"""Tests for runtime causal goal queuing, smooth mid-motion replacement, and smooth profiles."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest

from homunculus.backends.fake import FakeBackend
from homunculus.journal import Journal
from homunculus.models import (
    Authority,
    BaseVelocityGoal,
    GoalSpec,
    GoalSubmit,
    GripperWidthGoal,
    HandPostureGoal,
    JointTrajectoryGoal,
    RobotError,
    ServiceLimits,
    SessionOpen,
    StartConditions,
    TrajectoryPoint,
    ValidateRequest,
)
from homunculus.runtime import Backend, Clock, Principal, Runtime

CLIENT = Principal("client")
OTHER = Principal("other")
OPERATOR = Principal("operator", operator=True)
STEP = 0.02
PANDA_HOME = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": -1.57079,
    "joint5": 0.0,
    "joint6": 1.57079,
    "joint7": -0.7853,
}


class ManualClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t


class Harness:
    """One robot behind a runtime on a manual clock; leaving its `with` block closes journal and backend."""

    def __init__(
        self,
        journal_path: Path,
        robot: str,
        *,
        limits: ServiceLimits | None = None,
        make_backend: Callable[[Clock], Backend] | None = None,
    ):
        self.clock = ManualClock()
        self.journal_path = journal_path
        self.robot = robot
        self.limits = limits
        self.make_backend = make_backend
        self.requests = itertools.count(1)
        self.boots = 0
        self.incarnation = ExitStack()
        self.boot()

    def __enter__(self) -> Harness:
        return self

    def __exit__(self, *exc: object) -> None:
        self.incarnation.close()

    def boot(self) -> None:
        self.incarnation.close()
        self.boots += 1
        boot_id = f"boot-{self.boots:08d}"
        quota = (self.limits or ServiceLimits()).journal_quota
        with ExitStack() as opened:
            self.journal = opened.enter_context(Journal(self.journal_path, boot_id, quota=quota))
            if self.make_backend is None:
                backend, robot_id = FakeBackend(self.robot, self.clock), f"fake-{self.robot}"
            else:
                backend, robot_id = self.make_backend(self.clock), self.robot
            self.backend = opened.enter_context(backend)
            self.runtime = Runtime(
                robot_id, self.backend, self.journal, boot_id=boot_id, clock=self.clock, limits=self.limits
            )
            self.incarnation = opened.pop_all()
        self.context = self.runtime.describe().context_id
        self.session: str | None = None

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / STEP)):
            self.clock.t += STEP
            self.runtime.tick()
            self.journal.flush()

    def resource(self, rid: str):
        return next(r for r in self.runtime.describe().resources if r.id == rid)

    def open(self, principal: Principal = CLIENT) -> str:
        request = SessionOpen(request_id=f"open-{next(self.requests)}", context_id=self.context)
        self.session = self.runtime.open_session(principal, request).session_id
        return self.session

    def submit(self, action, *, request_id: str | None = None, principal: Principal = CLIENT, **settings):
        request = GoalSubmit(
            request_id=request_id or f"goal-{next(self.requests)}",
            session_id=settings.pop("session_id", self.session),
            context_id=settings.pop("context_id", self.context),
            action=action,
            **settings,
        )
        return self.runtime.submit_goal(principal, request)


def move_joint1(target: float, at: float = 1.0, *, start: float | None = None) -> JointTrajectoryGoal:
    points = [TrajectoryPoint(time_s=at, positions=[target], velocities=[0.0])]
    if start is not None:
        points.insert(0, TrajectoryPoint(time_s=0.0, positions=[start], velocities=[0.0]))
    return JointTrajectoryGoal(profile="joints.trajectory/1", target="arm", joints=["joint1"], points=points)


@pytest.fixture
def arm(tmp_path: Path) -> Iterator[Harness]:
    with Harness(tmp_path / "arm.sqlite", "arm") as harness:
        harness.open()
        yield harness


@pytest.fixture
def g1(tmp_path: Path) -> Iterator[Harness]:
    with Harness(tmp_path / "g1.sqlite", "g1") as harness:
        harness.open()
        yield harness


# --------------------------------------------------------------------------
# Causal goal queuing (after_goal_id)
# --------------------------------------------------------------------------


def test_default_without_after_goal_id_still_refuses_busy_resource(arm: Harness):
    g1 = arm.submit(move_joint1(0.4, at=1.0))
    arm.run(0.1)
    assert arm.resource("arm").activity == g1.goal_id
    assert arm.resource("arm").queued == []
    with pytest.raises(RobotError) as refused:
        arm.submit(move_joint1(0.8, at=1.0))
    assert refused.value.code == "resource_busy"


def test_same_resource_causal_chain_executes_back_to_back(arm: Harness):
    g1 = arm.submit(move_joint1(0.4, at=1.0))
    arm.run(0.1)
    g2 = arm.submit(move_joint1(0.8, at=1.0), after_goal_id=g1.goal_id)
    g3 = arm.submit(move_joint1(0.2, at=1.2), after_goal_id=g2.goal_id)

    status = arm.resource("arm")
    assert status.activity == g1.goal_id
    assert status.queued == [g2.goal_id, g3.goal_id]
    assert g2.state == "accepted"
    assert g3.state == "accepted"

    arm.run(5.0)
    for goal in (g1, g2, g3):
        assert arm.runtime.goal(goal.goal_id).state == "succeeded"
    assert arm.resource("arm").queued == []
    applied = [effect[1] for effect in arm.backend.effects if effect[0] == "apply"]
    assert applied == [g1.goal_id, g2.goal_id, g3.goal_id]
    obs = arm.runtime.observe()
    assert math.isclose(obs.samples["arm"].values["positions"]["joint1"], 0.2, abs_tol=0.01)


def test_cross_resource_chain_and_non_concurrent_chain(arm: Harness, g1: Harness):
    # arm -> gripper -> arm: G3 waits for gripper G2, which waits for arm G1.
    move1 = arm.submit(move_joint1(0.4, at=0.8))
    grip = arm.submit(
        GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=0.02), after_goal_id=move1.goal_id
    )
    move2 = arm.submit(move_joint1(0.7, at=0.8), after_goal_id=grip.goal_id)

    arm.run(0.2)
    assert arm.runtime.goal(move1.goal_id).state == "executing"
    assert arm.runtime.goal(grip.goal_id).state == "accepted"
    assert arm.runtime.goal(move2.goal_id).state == "accepted"
    assert arm.resource("gripper").activity is None
    assert arm.resource("gripper").queued == [grip.goal_id]

    arm.run(4.0)
    assert arm.runtime.goal(move1.goal_id).state == "succeeded"
    assert arm.runtime.goal(grip.goal_id).state == "succeeded"
    assert arm.runtime.goal(move2.goal_id).state == "succeeded"
    applied = [(effect[1], effect[3]) for effect in arm.backend.effects if effect[0] == "apply"]
    assert applied == [(move1.goal_id, "arm"), (grip.goal_id, "gripper"), (move2.goal_id, "arm")]

    # Non-concurrent resources on fake:g1 (base and left_hand cannot run together) can still be chained causally.
    drive = g1.submit(
        BaseVelocityGoal(
            profile="base.planar_velocity/1",
            target="base",
            forward_m_s=0.2,
            left_m_s=0.0,
            yaw_rate_rad_s=0.0,
            duration_s=0.8,
        )
    )
    hand = g1.submit(
        HandPostureGoal(profile="hand.posture/1", target="left_hand", posture="pinch"), after_goal_id=drive.goal_id
    )
    g1.run(0.2)
    assert g1.runtime.goal(drive.goal_id).state == "executing"
    assert g1.runtime.goal(hand.goal_id).state == "accepted"
    g1.run(3.0)
    assert g1.runtime.goal(drive.goal_id).state == "succeeded"
    assert g1.runtime.goal(hand.goal_id).state == "succeeded"


def test_parallel_fanout_and_conflicting_sibling_refusal(arm: Harness):
    g1 = arm.submit(move_joint1(0.3, at=0.8))
    arm.run(0.1)
    g2_arm = arm.submit(move_joint1(0.6, at=0.8), after_goal_id=g1.goal_id)
    g2_grip = arm.submit(
        GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=0.02), after_goal_id=g1.goal_id
    )

    # A second sibling on arm behind g1 conflicts with g2_arm.
    with pytest.raises(RobotError) as sibling:
        arm.submit(move_joint1(0.5, at=0.8), after_goal_id=g1.goal_id)
    assert sibling.value.code == "resource_busy"
    assert sibling.value.details["activity"] == g2_arm.goal_id

    # An immediate goal on gripper cannot jump ahead of g2_grip.
    with pytest.raises(RobotError) as jump:
        arm.submit(GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=0.01))
    assert jump.value.code == "resource_busy"
    assert jump.value.details["activity"] == g2_grip.goal_id

    arm.run(3.0)
    assert arm.runtime.goal(g1.goal_id).state == "succeeded"
    assert arm.runtime.goal(g2_arm.goal_id).state == "succeeded"
    assert arm.runtime.goal(g2_grip.goal_id).state == "succeeded"


def test_after_already_succeeded_predecessor_starts_immediately_even_after_trim(tmp_path: Path):
    with Harness(tmp_path / "journal.sqlite", "arm", limits=ServiceLimits(operation_retention=2)) as arm:
        arm.open()
        first = arm.submit(move_joint1(0.2, at=0.6))
        arm.run(1.5)
        assert arm.runtime.goal(first.goal_id).state == "succeeded"
        # Push `first` out of in-memory retention so lookup must consult the journal.
        for target in (0.3, 0.4, 0.5):
            goal = arm.submit(move_joint1(target, at=0.6))
            arm.run(1.5)
            assert arm.runtime.goal(goal.goal_id).state == "succeeded"
        followup = arm.submit(move_joint1(0.2, at=0.6), after_goal_id=first.goal_id)
        arm.run(1.5)
        assert arm.runtime.goal(followup.goal_id).state == "succeeded"


def test_predecessor_failure_cascade_cancels_queued_tail(arm: Harness):
    g1 = arm.submit(move_joint1(0.5, at=1.0))
    arm.run(0.1)
    g2 = arm.submit(move_joint1(0.9, at=1.0), after_goal_id=g1.goal_id)
    g3 = arm.submit(
        GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=0.02), after_goal_id=g2.goal_id
    )

    arm.backend.faults.add(("blocked", "arm"))
    arm.run(1.5)

    assert arm.runtime.goal(g1.goal_id).state == "failed"
    for tail, parent in ((g2, g1), (g3, g2)):
        done = arm.runtime.goal(tail.goal_id)
        assert done.state == "cancelled"
        assert done.result.reason == "predecessor_failed"
        assert done.result.measurements["predecessor_goal_id"] == parent.goal_id
    applied = [effect[1] for effect in arm.backend.effects if effect[0] == "apply"]
    assert applied == [g1.goal_id]

    # Submitting after a failed predecessor is refused at admission.
    with pytest.raises(RobotError) as refused:
        arm.submit(move_joint1(0.3, at=1.0), after_goal_id=g1.goal_id)
    assert refused.value.code == "start_condition_failed"
    assert refused.value.details["predecessor_state"] == "failed"


def test_cancel_middle_queued_goal_leaves_active_goal_running(arm: Harness):
    g1 = arm.submit(move_joint1(0.4, at=1.0))
    arm.run(0.1)
    g2 = arm.submit(move_joint1(0.8, at=1.0), after_goal_id=g1.goal_id)
    g3 = arm.submit(move_joint1(0.2, at=1.0), after_goal_id=g2.goal_id)

    cancelled = arm.runtime.cancel_goal(CLIENT, g2.goal_id, Authority(session_id=arm.session, context_id=arm.context))
    assert cancelled.state == "cancelled"
    assert cancelled.result.reason == "cancelled"
    assert arm.runtime.goal(g3.goal_id).state == "cancelled"
    assert arm.runtime.goal(g3.goal_id).result.reason == "predecessor_failed"
    assert arm.runtime.goal(g1.goal_id).state == "executing"

    arm.run(2.0)
    assert arm.runtime.goal(g1.goal_id).state == "succeeded"
    assert [e[1] for e in arm.backend.effects if e[0] == "apply"] == [g1.goal_id]


@pytest.mark.parametrize("action_kind", ["cancel_head", "stop", "close_session"])
def test_stopping_or_closing_session_ends_queued_goals_cleanly(arm: Harness, action_kind: str):
    g1 = arm.submit(move_joint1(0.4, at=1.0))
    arm.run(0.1)
    g2 = arm.submit(move_joint1(0.8, at=1.0), after_goal_id=g1.goal_id)

    if action_kind == "cancel_head":
        arm.runtime.cancel_goal(CLIENT, g1.goal_id, Authority(session_id=arm.session, context_id=arm.context))
        expected_reason = "predecessor_failed"
    elif action_kind == "stop":
        arm.runtime.stop(CLIENT, Authority(session_id=arm.session, context_id=arm.context))
        expected_reason = "stopped"
    else:
        arm.runtime.close_session(CLIENT, arm.session)
        expected_reason = "session_closed"

    arm.run(1.0)
    assert arm.runtime.goal(g1.goal_id).state == "cancelled"
    done2 = arm.runtime.goal(g2.goal_id)
    assert done2.state == "cancelled"
    assert done2.result.reason == expected_reason
    assert [e[1] for e in arm.backend.effects if e[0] == "apply"] == [g1.goal_id]


def test_validate_and_admission_use_expected_end_lookahead(arm: Harness):
    g1 = arm.submit(move_joint1(0.5, at=1.0))
    arm.run(0.1)

    # Valid follow-up from g1's expected end (0.5 -> 1.0 in 1.0 s) passes validate().
    valid = arm.runtime.validate(
        ValidateRequest(goal=GoalSpec(action=move_joint1(1.0, at=1.0), after_goal_id=g1.goal_id))
    )
    assert valid.valid is True
    assert valid.contract.after_goal_id == g1.goal_id

    # Impossible speed from expected end (0.5 -> 1.0 in 0.05 s) is refused at validate and admission.
    too_fast = move_joint1(1.0, at=0.05)
    with pytest.raises(RobotError) as fast_err:
        arm.runtime.validate(ValidateRequest(goal=GoalSpec(action=too_fast, after_goal_id=g1.goal_id)))
    assert fast_err.value.code == "limit_exceeded"

    # Explicit time-0 waypoint at 0.0 disagrees with g1's expected end at 0.5 -> refused early.
    wrong_start = move_joint1(0.8, at=1.0, start=0.0)
    with pytest.raises(RobotError) as start_err:
        arm.submit(wrong_start, after_goal_id=g1.goal_id)
    assert start_err.value.code == "start_condition_failed"


def test_dispatch_rechecks_real_measurements_not_expected_end(arm: Harness):
    obs = arm.runtime.observe()
    g1 = arm.submit(move_joint1(0.4, at=0.8))
    arm.run(0.1)
    g2 = arm.submit(
        move_joint1(0.8, at=0.8, start=0.4),
        after_goal_id=g1.goal_id,
        start_conditions=StartConditions(observation_id=obs.observation_id, max_position_change=0.45),
    )
    # Advance tick-by-tick until the tick g1 finishes: g2 has queued its dispatch write, not applied yet.
    while arm.runtime.goal(g1.goal_id).state != "succeeded":
        arm.run(STEP)
    assert arm.runtime.goal(g2.goal_id).state == "accepted"
    # Displace the real measured joint beyond start_position_tolerance before g2 dispatches.
    arm.backend._groups["arm"].positions["joint1"] = 0.55
    arm.run(0.2)
    done2 = arm.runtime.goal(g2.goal_id)
    assert done2.state == "failed"
    assert done2.result.reason == "precondition_changed"
    assert [e[1] for e in arm.backend.effects if e[0] == "apply"] == [g1.goal_id]


def test_queued_deadlines_and_max_queued_goals(tmp_path: Path):
    with Harness(tmp_path / "journal.sqlite", "arm", limits=ServiceLimits(max_queued_goals=2)) as arm:
        arm.open()
        # Execution deadline starts when head-of-line: three 1.0 s moves each with max_execution_s=1.8 succeed!
        g1 = arm.submit(move_joint1(0.2, at=1.0), max_execution_s=1.8, max_wall_s=10.0)
        g2 = arm.submit(move_joint1(0.4, at=1.0), after_goal_id=g1.goal_id, max_execution_s=1.8, max_wall_s=10.0)
        g3 = arm.submit(move_joint1(0.6, at=1.0), after_goal_id=g2.goal_id, max_execution_s=1.8, max_wall_s=10.0)
        # Queue is now full (2 queued: g2 and g3).
        with pytest.raises(RobotError) as full:
            arm.submit(move_joint1(0.8, at=1.0), after_goal_id=g3.goal_id, max_execution_s=1.8, max_wall_s=10.0)
        assert full.value.code == "limit_exceeded"
        assert full.value.details["parameter"] == "max_queued_goals"

        arm.run(5.0)
        assert arm.runtime.goal(g1.goal_id).state == "succeeded"
        assert arm.runtime.goal(g2.goal_id).state == "succeeded"
        assert arm.runtime.goal(g3.goal_id).state == "succeeded"

        # Wall deadline starts at admission: if a queued goal expires while waiting, it fails without stopping head.
        h1 = arm.submit(move_joint1(0.1, at=2.0), max_wall_s=5.0)
        arm.run(0.1)
        h2 = arm.submit(move_joint1(0.3, at=0.5), after_goal_id=h1.goal_id, max_wall_s=1.0)
        h3 = arm.submit(move_joint1(0.5, at=0.5), after_goal_id=h2.goal_id, max_wall_s=5.0)
        arm.run(3.0)
        assert arm.runtime.goal(h1.goal_id).state == "succeeded"
        assert arm.runtime.goal(h2.goal_id).state == "failed"
        assert arm.runtime.goal(h2.goal_id).result.reason == "precondition_changed"
        assert arm.runtime.goal(h3.goal_id).state == "cancelled"
        assert arm.runtime.goal(h3.goal_id).result.reason == "predecessor_failed"


def test_queued_idempotency_and_restart_recovery(arm: Harness):
    g1 = arm.submit(move_joint1(0.4, at=1.0))
    arm.run(0.1)
    g2 = arm.submit(move_joint1(0.8, at=1.0), after_goal_id=g1.goal_id, request_id="pipe-2")
    replay = arm.submit(move_joint1(0.8, at=1.0), after_goal_id=g1.goal_id, request_id="pipe-2")
    assert replay.goal_id == g2.goal_id
    with pytest.raises(RobotError) as conflict:
        arm.submit(move_joint1(0.9, at=1.0), after_goal_id=g1.goal_id, request_id="pipe-2")
    assert conflict.value.code == "request_conflict"

    # Crash/restart while g1 is executing and g2 is queued: g1 is uncertain, g2 is truthfully not_dispatched.
    arm.boot()
    after_g1 = arm.runtime.goal(g1.goal_id)
    after_g2 = arm.runtime.goal(g2.goal_id)
    assert after_g1.state == "interrupted"
    assert after_g1.result.measurements["dispatch"] == "uncertain"
    assert after_g2.state == "interrupted"
    assert after_g2.result.measurements["dispatch"] == "not_dispatched"


def test_cross_session_and_unknown_predecessor_refused(arm: Harness):
    g1 = arm.submit(move_joint1(0.4, at=1.0))
    with pytest.raises(RobotError) as unknown:
        arm.submit(move_joint1(0.8, at=1.0), after_goal_id="goal-does-not-exist")
    assert unknown.value.code == "start_condition_failed"

    arm.run(1.5)
    assert arm.runtime.goal(g1.goal_id).state == "succeeded"
    arm.runtime.close_session(CLIENT, arm.session)
    other_session = arm.open(OTHER)
    with pytest.raises(RobotError) as cross_session:
        arm.submit(move_joint1(0.8, at=1.0), after_goal_id=g1.goal_id, principal=OTHER, session_id=other_session)
    assert cross_session.value.code == "permission_denied"
