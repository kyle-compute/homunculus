"""Behavioural checks for the shared runtime, on the fake backend and a manual clock.

Each test asserts what a client would see and what the backend was told to do
(``backend.effects``), not how the runtime arrived there. Passing here is
evidence about the protocol; it says nothing about physics or a real robot.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import itertools
import json
import math
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Literal

import anyio.to_thread
import httpx
import pytest
import uvicorn
from pydantic import Field, ValidationError

from homunculus import journal, rigs, server
from homunculus.backends.fake import FakeBackend
from homunculus.backends.task import TaskBackend, _start_simulator
from homunculus.backends.task import robots as task_robots
from homunculus.cli import main as cli_main
from homunculus.client import ClientError, RobotClient, Target, TargetStore
from homunculus.grasp import GripperGraspGoal
from homunculus.journal import Journal, JournalInUse, held
from homunculus.models import (
    Authority,
    BaseVelocityGoal,
    BaseVelocityLimits,
    BaseVelocityTarget,
    CameraImage,
    GoalSpec,
    GoalSubmit,
    GripperWidthGoal,
    HandPostureGoal,
    JointTrajectoryGoal,
    ManagementRequest,
    Model,
    Observation,
    Pose,
    ProfileBinding,
    RobotError,
    SensorInfo,
    ServiceLimits,
    SessionOpen,
    StartConditions,
    StreamOpen,
    StreamUpdate,
    ToolPoseGoal,
    TrajectoryPoint,
    ValidateRequest,
    VideoLink,
)
from homunculus.profiles import (
    PROFILES,
    ActivityStatus,
    MotionJudge,
    Profile,
    VelocityJudge,
    measures,
    register,
    sample_trajectory,
)
from homunculus.runtime import Backend, BackendFacts, Clock, Frame, Principal, Runtime, StartupError
from homunculus.server import Service, create_app, load_backends, serve

CLIENT = Principal("client")
OTHER = Principal("other")
OPERATOR = Principal("operator", operator=True)
STEP = 0.02
ARM_JOINTS = [f"joint{n}" for n in range(1, 8)]  # the Franka Emika Panda's, on the fake arm and in MuJoCo
PANDA_HOME = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": -1.57079,
    "joint5": 0.0,
    "joint6": 1.57079,
    "joint7": -0.7853,
}
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
EXAMPLES = Path(__file__).parent.parent / "examples"


class ManualClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t


class Harness:
    """One robot behind a runtime, on a clock that only the test advances: a fake one unless `backend` is given.

    It holds the journal and the robot of the incarnation that is running. Leaving its `with` block closes both.
    """

    def __init__(
        self,
        journal_path,
        robot: str,
        limits: ServiceLimits | None = None,
        backend: Callable[[Clock], Backend] | None = None,
    ):
        self.clock = ManualClock()
        self.journal_path = journal_path
        self.robot = robot
        self.limits = limits
        self.make_backend = backend
        self.requests = itertools.count(1)
        self.boots = 0
        self.incarnation = ExitStack()  # the journal and the robot that the running incarnation holds
        self.boot()

    def __enter__(self) -> Harness:
        return self

    def __exit__(self, *exc: object) -> None:
        self.incarnation.close()

    def boot(self) -> None:
        """Start a runtime incarnation as a process restart would: new boot, same journal file."""
        self.incarnation.close()  # the process that held the journal and the robot is gone
        self.boots += 1
        boot_id = f"boot-{self.boots:08d}"
        quota = (self.limits or ServiceLimits()).journal_quota
        with ExitStack() as opened:  # closed again if the runtime refuses the robot
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
        self.session = None

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / STEP)):
            self.clock.t += STEP
            self.runtime.tick()
            self.journal.flush()

    def effects(self, kind: str) -> list[tuple]:
        return [effect for effect in self.backend.effects if effect[0] == kind]

    def resource(self, rid: str):
        return next(r for r in self.runtime.describe().resources if r.id == rid)

    def open(self, principal: Principal = CLIENT, **settings):
        request = SessionOpen(request_id=f"open-{next(self.requests)}", context_id=self.context, **settings)
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

    def finish(self, goal, seconds: float):
        self.run(seconds)
        return self.runtime.goal(goal.goal_id)

    def manage(self, operation: str, seconds: float = 1.0):
        request = ManagementRequest(
            request_id=f"manage-{next(self.requests)}", context_id=self.context, operation=operation
        )
        started = self.runtime.manage(OPERATOR, request)
        self.run(seconds)
        return self.runtime.operation(started.operation_id)

    def open_stream(self, forward: float, validity: float = 0.5, basis: str | None = None):
        request = StreamOpen(
            request_id=f"stream-{next(self.requests)}",
            session_id=self.session,
            context_id=self.context,
            profile="base.planar_velocity/1",
            target="base",
            update_validity_wall_s=validity,
            initial_target=velocity(forward),
            basis_observation_id=basis or self.runtime.observe().observation_id,
        )
        return self.runtime.open_stream(CLIENT, request)

    def update(self, stream, sequence: int, forward: float, *, basis: str | None = None, principal=CLIENT):
        request = StreamUpdate(
            session_id=self.session,
            context_id=self.context,
            sequence=sequence,
            basis_observation_id=basis or self.runtime.observe().observation_id,
            target=velocity(forward),
        )
        return self.runtime.update_stream(principal, stream.stream_id, request)


@pytest.fixture
def arm(tmp_path) -> Iterator[Harness]:
    with Harness(tmp_path / "journal.sqlite", "arm") as harness:
        yield harness


@pytest.fixture
def g1(tmp_path) -> Iterator[Harness]:
    with Harness(tmp_path / "journal.sqlite", "g1") as harness:
        yield harness


def velocity(forward: float) -> BaseVelocityTarget:
    return BaseVelocityTarget(forward_m_s=forward, left_m_s=0.0, yaw_rate_rad_s=0.0)


def drive(forward: float, duration: float = 1.0) -> BaseVelocityGoal:
    return BaseVelocityGoal(
        profile="base.planar_velocity/1",
        target="base",
        forward_m_s=forward,
        left_m_s=0.0,
        yaw_rate_rad_s=0.0,
        duration_s=duration,
    )


def reach(at: float = 1.0, **positions: float) -> JointTrajectoryGoal:
    """Arm joints to positions, reached `at` seconds from now; the other joints hold."""
    point = TrajectoryPoint(time_s=at, positions=list(positions.values()), velocities=[0.0] * len(positions))
    return JointTrajectoryGoal(profile="joints.trajectory/1", target="arm", joints=list(positions), points=[point])


def gripper(width: float, **settings) -> GripperWidthGoal:
    return GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=width, **settings)


def posture(name: str) -> HandPostureGoal:
    return HandPostureGoal(profile="hand.posture/1", target="left_hand", posture=name)


def refused(call) -> str:
    """The stable error code of a request the runtime must reject."""
    with pytest.raises(RobotError) as rejection:
        call()
    return rejection.value.code


# --------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------


def test_a_repeated_request_is_one_operation_executed_once(arm):
    session = arm.open()
    first = arm.submit(gripper(0.03), request_id="close-jaw")
    before_dispatch = arm.submit(gripper(0.03), request_id="close-jaw")
    assert before_dispatch.goal_id == first.goal_id

    assert arm.finish(first, 1.0).state == "succeeded"
    after_completion = arm.submit(gripper(0.03), request_id="close-jaw")
    assert (after_completion.goal_id, after_completion.state) == (first.goal_id, "succeeded")
    assert len(arm.effects("apply")) == 1

    # The same identity with different content is a conflict, not a second command.
    assert refused(lambda: arm.submit(gripper(0.05), request_id="close-jaw")) == "request_conflict"
    assert len(arm.effects("apply")) == 1

    # Opening a session is retried the same way: one session, not a conflict with itself.
    request = SessionOpen(request_id="open-1", context_id=arm.context)
    assert arm.runtime.open_session(CLIENT, request).session_id == session


# --------------------------------------------------------------------------
# Stale authority and stale state
# --------------------------------------------------------------------------


# Each case prepares a situation and returns the request that must be refused.


def wrong_context(arm: Harness):
    arm.open()
    return lambda: arm.submit(gripper(0.03), context_id="ctx-000000000000")


def expired_session(arm: Harness):
    arm.open(lifetime_wall_s=1.0)
    arm.run(1.5)
    return lambda: arm.submit(gripper(0.03))


def closed_session(arm: Harness):
    arm.runtime.close_session(CLIENT, arm.open())
    return lambda: arm.submit(gripper(0.03))


def another_principal(arm: Harness):
    arm.open()
    return lambda: arm.submit(gripper(0.03), principal=OTHER)


def state_moved_since_observation(arm: Harness):
    arm.open()
    seen = arm.runtime.observe()
    assert arm.finish(arm.submit(gripper(0.03)), 1.0).state == "succeeded"
    conditions = StartConditions(observation_id=seen.observation_id, max_position_change=0.01)
    return lambda: arm.submit(gripper(0.06), start_conditions=conditions)


def feedback_lost(arm: Harness):
    arm.open()
    arm.backend.faults.add(("feedback_loss", "gripper"))
    arm.run(STEP)
    return lambda: arm.submit(gripper(0.03))


@pytest.mark.parametrize(
    "prepare, code",
    [
        (wrong_context, "stale_context"),
        (expired_session, "session_invalid"),
        (closed_session, "session_invalid"),
        (another_principal, "permission_denied"),
        (state_moved_since_observation, "start_condition_failed"),
        (feedback_lost, "start_condition_failed"),
    ],
)
def test_stale_authority_or_state_is_rejected_before_any_movement(arm, prepare, code):
    request = prepare(arm)
    commanded = list(arm.backend.effects)
    assert refused(request) == code
    arm.run(0.5)
    assert arm.backend.effects == commanded
    assert arm.resource("gripper").activity is None


def test_start_conditions_are_rechecked_at_dispatch(arm):
    arm.open()
    conditions = StartConditions(max_measurement_age_s=0.5)
    goal = arm.submit(reach(joint1=0.4), start_conditions=conditions)
    assert goal.state == "accepted"

    # Feedback disappears after acceptance and before the controller picks the goal up.
    arm.backend.faults.add(("feedback_loss", "arm"))
    goal = arm.finish(goal, 0.1)
    assert (goal.state, goal.result.reason) == ("failed", "precondition_changed")
    assert goal.activated_time_ns is None
    assert arm.effects("apply") == []
    assert arm.resource("arm").activity is None


def test_authority_lost_before_dispatch_cancels_without_movement(arm):
    arm.open(lifetime_wall_s=1.0)
    goal = arm.submit(gripper(0.03))
    arm.clock.t += 2.0  # the session expires before the next controller tick
    goal = arm.finish(goal, 0.1)
    assert (goal.state, goal.result.reason) == ("failed", "session_expired")
    assert arm.effects("apply") == []


def test_a_goal_past_its_deadline_neither_starts_nor_counts_as_completed(arm):
    arm.open()
    late = arm.submit(gripper(0.03), max_execution_s=0.5)
    arm.clock.t += 1.0  # the control loop stalls between admission and dispatch
    late = arm.finish(late, 0.1)
    assert (late.state, late.result.reason) == ("failed", "precondition_changed")
    assert arm.effects("apply") == []

    # It stalls during a movement, and the jaw has arrived by the next reading. Completion first measured
    # after the deadline is not success within it: the goal ends through a confirmed stop, as failed.
    goal = arm.submit(gripper(0.03), max_execution_s=0.5)
    arm.run(0.1)
    arm.clock.t += 5.0
    goal = arm.finish(goal, 0.5)
    assert (goal.state, goal.result.reason, goal.result.stop_confirmed) == (
        "failed",
        "execution_deadline_exceeded",
        True,
    )


@pytest.mark.parametrize(
    ("robot", "action", "too_short", "enough"),
    [
        # The longest trajectory the arm advertises, under the default deadlines.
        ("arm", reach(joint1=0.2, at=30.0), {}, {"max_execution_s": 31.0}),
        ("arm", reach(joint1=0.2), {"max_wall_s": 1.1}, {"max_wall_s": 1.5}),
        ("g1", drive(0.3, 2.0), {"max_execution_s": 2.1}, {"max_execution_s": 2.5}),
    ],
)
def test_a_goal_that_cannot_finish_within_its_deadline_is_refused_before_it_moves(
    tmp_path, robot, action, too_short, enough
):
    with Harness(tmp_path / "journal.sqlite", robot) as harness:
        harness.open()

        def dry_run(**settings):
            return harness.runtime.validate(ValidateRequest(goal=GoalSpec(action=action, **settings)))

        # The motion a request states, and the rest it must then be measured at, end after the deadline does.
        assert refused(lambda: harness.submit(action, **too_short)) == "limit_exceeded"
        # A dry run checks the goal as it would be submitted, its deadlines and start conditions included.
        assert refused(lambda: dry_run(**too_short)) == "limit_exceeded"
        elsewhere = StartConditions(required_mode="another mode")
        assert refused(lambda: dry_run(**enough, start_conditions=elsewhere)) == "start_condition_failed"
        assert dry_run(**enough).valid
        assert harness.backend.effects == []
        assert all(resource.activity is None for resource in harness.runtime.describe().resources)

        goal = harness.finish(harness.submit(action, **enough), 31.5)
        assert (goal.state, goal.result.reason) == ("succeeded", "completed")


# --------------------------------------------------------------------------
# Target streams
# --------------------------------------------------------------------------


def test_stream_targets_replace_in_order_and_only_the_newest_is_applied(g1):
    g1.open()
    stream = g1.open_stream(forward=0.2)
    g1.run(0.1)
    assert g1.runtime.stream(stream.stream_id).applied_sequence == 0

    # Two updates land between controller ticks; the superseded one is never dispatched.
    g1.update(stream, 1, forward=0.3)
    basis = g1.runtime.observe().observation_id
    newest = g1.update(stream, 2, forward=0.4, basis=basis)
    g1.run(STEP)
    assert [effect[2] for effect in g1.effects("track")] == [0.2, 0.4]
    assert g1.runtime.stream(stream.stream_id).applied_sequence == 2
    assert g1.effects("stop") == []  # replaced without stopping in between

    # A retransmission returns the first acknowledgement and does not extend validity.
    g1.run(0.1)
    duplicate = g1.update(stream, 2, forward=0.4, basis=basis)
    assert duplicate.duplicate and duplicate.ack == newest.ack
    assert duplicate.stream.expires_at_ns == newest.stream.expires_at_ns

    assert refused(lambda: g1.update(stream, 2, forward=0.1, basis=basis)) == "sequence_conflict"
    g1.update(stream, 5, forward=0.4)
    assert refused(lambda: g1.update(stream, 4, forward=0.1)) == "sequence_rejected"
    assert (
        refused(lambda: g1.update(stream, 6, forward=0.4, basis=basis)) == "basis_invalid"
    )  # older than the last basis
    assert refused(lambda: g1.update(stream, 6, forward=0.9)) == "limit_exceeded"
    assert refused(lambda: g1.update(stream, 6, forward=0.4, principal=OTHER)) == "permission_denied"

    g1.run(STEP)
    assert [effect[2] for effect in g1.effects("track")] == [0.2, 0.4, 0.4]
    assert g1.runtime.observe().samples["base"].values["velocities"]["forward_m_s"] == 0.4


def test_a_target_is_only_valid_for_its_window_after_the_basis_observation(g1):
    g1.open()
    stream = g1.open_stream(forward=0.2, validity=0.5)
    old_basis = g1.runtime.observe().observation_id
    g1.run(0.3)
    g1.update(stream, 1, forward=0.2)  # a fresh basis keeps the stream alive
    g1.run(0.3)
    # 0.6 s after it was issued, the old observation cannot justify a new target.
    assert refused(lambda: g1.update(stream, 2, forward=0.3, basis=old_basis)) == "basis_invalid"

    # With no fresh target the last one runs out: the base is stopped and the stream closes itself.
    g1.run(0.6)
    closed = g1.runtime.stream(stream.stream_id)
    assert (closed.state, closed.closure) == ("closed", "expired")
    assert g1.effects("stop") == [("stop", ("base",))]
    assert g1.runtime.observe().samples["base"].values["velocities"]["forward_m_s"] == 0.0
    assert g1.resource("base").condition == "ready"

    # A late packet cannot reopen it.
    tracked = len(g1.effects("track"))
    assert refused(lambda: g1.update(stream, 2, forward=0.3)) == "stream_closed"
    g1.run(0.1)
    assert len(g1.effects("track")) == tracked


def test_a_stream_cannot_open_on_an_observation_older_than_its_window(g1):
    g1.open()
    basis = g1.runtime.observe().observation_id
    g1.run(0.6)
    assert refused(lambda: g1.open_stream(forward=0.2, validity=0.5, basis=basis)) == "target_expired"
    assert g1.effects("track") == [] and g1.resource("base").activity is None


def test_a_stream_target_is_not_applied_without_valid_measurements(g1):
    g1.open()
    stream = g1.open_stream(forward=0.3)
    g1.backend.faults.add(("feedback_loss", "base"))  # after admission, before the controller tick
    g1.run(0.1)
    stream = g1.runtime.stream(stream.stream_id)
    assert (stream.state, stream.closure, stream.detail, stream.applied_sequence) == (
        "closed",
        "failed",
        "feedback_lost",
        None,
    )
    assert g1.effects("track") == [] and g1.resource("base").activity is None
    # A fresh observation ID does not help while the base is still unmeasured.
    assert refused(lambda: g1.open_stream(forward=0.3)) == "start_condition_failed"


# --------------------------------------------------------------------------
# What stays after a movement
# --------------------------------------------------------------------------


def test_a_finished_goal_leaves_its_hold_and_the_hold_outlives_the_session(arm):
    arm.open()
    goal = arm.finish(arm.submit(reach(joint1=0.5, joint2=-0.3, joint4=-0.8)), 1.5)
    assert (goal.state, goal.result.reason) == ("succeeded", "completed")

    # Success is backed by a measurement, taken after the goal settled.
    measured = arm.runtime.observation(goal.result.completion_observation_id).samples["arm"].values["positions"]
    assert measured == pytest.approx({**PANDA_HOME, "joint1": 0.5, "joint2": -0.3, "joint4": -0.8})
    assert goal.retained.kind == "hold" and goal.retained.source == goal.goal_id
    assert goal.retained.target == pytest.approx(measured)

    # Ending the session revokes authority; it does not release what the controller holds.
    arm.runtime.close_session(CLIENT, arm.session)
    arm.run(0.5)
    assert arm.effects("stop") == []
    held = arm.resource("arm")
    assert (held.condition, held.retained.kind, held.retained.source) == ("ready", "hold", goal.goal_id)

    # An explicit stop is what replaces the hold with the mode's fallback.
    stop = arm.runtime.stop(OPERATOR, Authority())
    arm.run(0.5)
    assert arm.runtime.operation(stop.operation_id).state == "succeeded"
    assert arm.resource("arm").retained.kind == "fallback"


def test_cancelling_stops_only_that_goal_and_reports_a_confirmed_stop(arm):
    arm.open()
    moving = arm.submit(reach(joint1=1.0, at=2.0))
    jaw = arm.submit(gripper(0.0))  # the arm and gripper are validated to run together
    arm.run(0.4)
    authority = Authority(session_id=arm.session, context_id=arm.context)
    assert arm.runtime.cancel_goal(CLIENT, moving.goal_id, authority).state == "stopping"

    moving = arm.finish(moving, 0.5)
    assert (moving.state, moving.result.reason, moving.result.stop_confirmed) == ("cancelled", "cancelled", True)
    assert arm.effects("stop") == [("stop", ("arm",))]
    assert arm.finish(jaw, 0.5).state == "succeeded"
    # Cancelling again changes nothing.
    assert arm.runtime.cancel_goal(CLIENT, moving.goal_id, authority).state == "cancelled"
    assert len(arm.effects("stop")) == 1


def test_takeover_revokes_the_owner_and_grants_authority_only_after_its_motion_stopped(arm):
    old_session = arm.open()
    moving = arm.submit(reach(joint1=1.0, at=2.0))
    arm.run(0.5)
    takeover = lambda who, rid: arm.runtime.open_session(  # noqa: E731
        who, SessionOpen(request_id=rid, context_id=arm.context, takeover=True)
    )
    assert refused(lambda: takeover(OTHER, "take-1")) == "permission_denied"
    assert arm.runtime.goal(moving.goal_id).state == "executing"

    # The owner is revoked at once, but nobody may command while its motion is still stopping.
    assert refused(lambda: takeover(OPERATOR, "take-2")) == "resource_busy"
    assert refused(lambda: arm.submit(gripper(0.03))) == "session_invalid"
    moving = arm.finish(moving, 0.5)
    assert (moving.state, moving.result.reason, moving.result.stop_confirmed) == (
        "cancelled",
        "operator_takeover",
        True,
    )

    new_session = takeover(OPERATOR, "take-3").session_id
    walking = arm.submit(reach(joint1=0.5, at=2.0), principal=OPERATOR, session_id=new_session)
    arm.run(0.2)
    # The revoked client can neither command nor stop the new owner's motion.
    stale = Authority(session_id=old_session, context_id=arm.context)
    assert refused(lambda: arm.runtime.stop(CLIENT, stale)) == "session_invalid"
    assert refused(lambda: arm.runtime.cancel_goal(CLIENT, walking.goal_id, stale)) == "session_invalid"
    assert arm.finish(walking, 2.5).state == "succeeded"
    assert len(arm.effects("stop")) == 1

    # The operator's session ends and the client owns the robot again. The ended session is no authority,
    # for an operator either: only a stop that names no session acts outside one.
    arm.runtime.close_session(OPERATOR, new_session)
    arm.open()
    moving = arm.submit(reach(joint1=0.0, at=2.0))
    arm.run(0.2)
    ended = Authority(session_id=new_session, context_id=arm.context)
    assert refused(lambda: arm.runtime.stop(OPERATOR, ended)) == "session_invalid"
    assert refused(lambda: arm.runtime.cancel_goal(OPERATOR, moving.goal_id, ended)) == "session_invalid"
    assert arm.runtime.goal(moving.goal_id).state == "executing"
    arm.runtime.stop(OPERATOR, Authority())
    assert arm.finish(moving, 0.5).result.reason == "stopped"


def test_a_stop_is_sent_until_it_is_delivered_and_asking_again_does_not_restart_it(arm):
    arm.open()
    moving = arm.submit(reach(joint1=1.0, at=2.0))
    arm.run(0.5)
    arm.backend.faults.add(("stop_lost", "arm"))  # the link drops the first stop commands
    first = arm.runtime.stop(OPERATOR, Authority())
    arm.run(0.1)
    assert arm.effects("stop") == [] and arm.resource("arm").condition == "stopping"
    arm.backend.faults.clear()

    # An operator who keeps asking must not keep the stop from being confirmed: commanding a resource that
    # is already stopping would restart its standstill measurement each time, up to the stopping deadline.
    for _ in range(8):
        arm.run(0.1)
        arm.runtime.stop(OPERATOR, Authority())
    arm.run(0.5)
    moving = arm.runtime.goal(moving.goal_id)
    assert (moving.state, moving.result.reason, moving.result.stop_confirmed) == ("cancelled", "stopped", True)
    assert arm.runtime.operation(first.operation_id).state == "succeeded"
    assert arm.resource("arm").condition == "ready"


def test_a_lost_goal_acknowledgement_keeps_the_arm_reserved_until_it_stops(arm, monkeypatch):
    apply = arm.backend.apply

    def loses_reply(*args):
        apply(*args)
        raise ConnectionError("goal delivered without an acknowledgement")

    monkeypatch.setattr(arm.backend, "apply", loses_reply)
    arm.open()
    goal = arm.submit(reach(joint1=1.0, at=2.0))
    assert arm.finish(goal, 2 * STEP).state == "stopping"
    assert arm.resource("arm").activity == goal.goal_id
    assert arm.effects("stop") == [("stop", ("arm",))]
    assert refused(lambda: arm.submit(reach(joint1=0.5))) == "resource_busy"

    goal = arm.finish(goal, 0.5)
    assert (goal.state, goal.result.reason, goal.result.stop_confirmed) == ("failed", "backend_error", True)
    assert arm.runtime.observe().samples["arm"].values["positions"]["joint1"] == 0.0


def test_a_lost_first_stream_acknowledgement_stops_without_claiming_delivery(g1, monkeypatch):
    track = g1.backend.track

    def loses_reply(*args):
        track(*args)
        raise ConnectionError("target delivered without an acknowledgement")

    monkeypatch.setattr(g1.backend, "track", loses_reply)
    g1.open()
    stream = g1.open_stream(0.3)
    g1.run(2 * STEP)
    live = g1.runtime.stream(stream.stream_id)
    assert (live.state, live.applied_sequence) == ("stopping", None)
    assert g1.effects("stop") == [("stop", ("base",))]
    g1.run(0.5)
    closed = g1.runtime.stream(stream.stream_id)
    assert (closed.state, closed.closure, closed.applied_sequence) == ("closed", "failed", None)
    assert g1.runtime.observe().samples["base"].values["positions"]["x_m"] == 0.0


def test_a_command_is_never_sent_without_its_durable_dispatch_marker(arm, monkeypatch):
    record = arm.journal.record

    def refuse_marker(*args, finished):
        if not finished:
            raise OSError("injected journal write failure")
        record(*args, finished=finished)

    monkeypatch.setattr(arm.journal, "record", refuse_marker)
    arm.open()
    goal = arm.finish(arm.submit(reach(joint1=1.0, at=2.0)), 0.1)
    assert (goal.state, goal.result.reason, goal.activated_time_ns) == ("failed", "journal_error", None)
    assert arm.effects("apply") == arm.effects("stop") == []


def test_a_stop_is_sent_while_the_outcome_of_another_move_cannot_be_saved(g1, monkeypatch, caplog):
    # This driver supports base and hand work at the same time.
    g1.runtime._facts.concurrent = [["base", "left_hand"]]
    g1.open()
    stream = g1.open_stream(0.3)
    g1.run(0.1)
    queued = g1.submit(posture("fist"))
    writing, release = threading.Event(), threading.Event()
    record = g1.journal.record

    def held_write(robot_id, operation_id, document, *, finished):
        if operation_id == queued.goal_id:
            writing.set()
            assert release.wait(5)
            raise OSError("injected journal write failure")
        record(robot_id, operation_id, document, finished=finished)

    monkeypatch.setattr(g1.journal, "record", held_write)
    stop = g1.runtime.stop(OPERATOR, Authority())
    try:
        assert writing.wait(5)
        g1.clock.t += STEP
        g1.runtime.tick()
        assert g1.effects("stop") == [("stop", ("base", "left_hand"))]
    finally:
        release.set()
    g1.run(0.5)
    assert g1.runtime.operation(stop.operation_id).state == "succeeded"
    assert g1.runtime.stream(stream.stream_id).state == "closed"
    assert g1.runtime.goal(queued.goal_id).state == "cancelled"
    assert queued.goal_id in caplog.text and "injected journal write failure" in caplog.text


def test_a_stop_invalidates_a_move_whose_admission_write_is_still_in_progress(arm, monkeypatch):
    arm.open()
    writing, release = threading.Event(), threading.Event()
    admit = arm.journal.admit

    def held_write(**request):
        writing.set()
        assert release.wait(5)
        admit(**request)

    monkeypatch.setattr(arm.journal, "admit", held_write)
    with ThreadPoolExecutor() as worker:
        goal_reply = worker.submit(arm.submit, reach(joint1=1.0, at=2.0))
        try:
            assert writing.wait(5)
            stop = arm.runtime.stop(OPERATOR, Authority())
            arm.run(0.5)
            assert arm.runtime.operation(stop.operation_id).state == "succeeded"
        finally:
            release.set()
        goal = goal_reply.result(timeout=5)
    arm.run(0.5)
    assert (goal.state, goal.result.reason) == ("cancelled", "stopped")
    assert arm.effects("apply") == []


def test_shutdown_stops_the_other_robot_while_one_driver_is_stalled(arm, tmp_path, monkeypatch):
    with Harness(tmp_path / "second.sqlite", "g1") as g1:
        blocked, release, leave, stopped = (threading.Event() for _ in range(4))
        stop = g1.backend.stop

        def stalled_tick():
            blocked.set()
            assert release.wait(5)

        def report_stop(resources):
            stop(resources)
            stopped.set()

        monkeypatch.setattr(arm.runtime, "tick", stalled_tick)
        monkeypatch.setattr(g1.backend, "stop", report_stop)
        service = Service({"arm": arm.runtime, "g1": g1.runtime}, {}, "boot", ServiceLimits())

        def serve_until_requested():
            with service:
                assert leave.wait(5)

        with ThreadPoolExecutor() as worker:
            shutdown = worker.submit(serve_until_requested)
            try:
                assert blocked.wait(5)
                leave.set()
                assert stopped.wait(2)
                assert not shutdown.done()
            finally:
                leave.set()
                release.set()
            shutdown.result(timeout=5)


@pytest.mark.parametrize("device_cadence", [True, False])
def test_a_drivers_done_report_without_measured_completion_fails(arm, monkeypatch, device_cadence):
    arm.open()
    goal = arm.submit(reach(joint1=1.0, at=2.0))
    advance = arm.backend.advance

    def premature_completion():
        feedback = advance()
        feedback.activities[goal.goal_id] = ActivityStatus("done")
        return feedback

    monkeypatch.setattr(arm.backend, "reports_goal_samples", device_cadence)
    monkeypatch.setattr(arm.backend, "apply", lambda *_: None)
    monkeypatch.setattr(arm.backend, "advance", premature_completion)
    goal = arm.finish(goal, 0.5)
    assert (goal.state, goal.result.reason, goal.result.stop_confirmed) == ("failed", "completion_unverified", True)


# --------------------------------------------------------------------------
# Stopping that cannot be confirmed
# --------------------------------------------------------------------------


def test_an_unconfirmed_stop_blocks_motion_until_an_operator_recovers(arm):
    arm.open()
    goal = arm.submit(reach(joint1=1.0, at=2.0))
    arm.run(0.5)
    arm.backend.faults.add(("feedback_loss", "arm"))
    goal = arm.finish(goal, 1.0)

    # Lost feedback is not standstill: the stop was commanded but never confirmed.
    assert ("stop", ("arm",)) in arm.effects("stop")
    assert (goal.state, goal.result.reason, goal.result.stop_confirmed) == ("failed", "stop_unconfirmed", False)
    assert goal.result.measurements["cause"] == "feedback_lost"
    faulted = arm.resource("arm")
    assert (faulted.condition, faulted.fault, faulted.retained) == ("faulted", "stop_unconfirmed", None)

    applied = len(arm.effects("apply"))
    assert refused(lambda: arm.submit(reach(joint1=0.2))) == "resource_faulted"
    binding = next(p for p in arm.runtime.describe().profiles if p.target == "arm")
    assert not binding.available and "faulted" in binding.unavailable_reason
    # A later stop cannot report success around the resource that never showed standstill.
    stop = arm.runtime.stop(OPERATOR, Authority())
    arm.run(0.5)
    stop = arm.runtime.operation(stop.operation_id)
    assert (stop.state, stop.reason, stop.resources["arm"]) == ("failed", "stop_unconfirmed", "faulted")

    # Recovery is an operator action, and it needs measured standstill, not a cleared flag.
    recover = ManagementRequest(request_id="recover-as-client", context_id=arm.context, operation="recover")
    assert refused(lambda: arm.runtime.manage(CLIENT, recover)) == "permission_denied"
    failed = arm.manage("recover")
    assert (failed.state, failed.reason) == ("failed", "recovery_failed")
    arm.backend.faults.clear()
    arm.run(0.5)
    assert refused(lambda: arm.submit(reach(joint1=0.2))) == "resource_faulted"
    assert len(arm.effects("apply")) == applied

    recovered = arm.manage("recover")
    assert recovered.state == "succeeded" and recovered.new_context_id != arm.context
    # Authority starts over: the old session and context are no longer accepted.
    assert refused(lambda: arm.submit(reach(joint1=0.2))) == "stale_context"
    arm.context = recovered.new_context_id
    arm.open()
    assert arm.finish(arm.submit(reach(joint1=0.2)), 1.5).state == "succeeded"


def test_a_recovery_that_restores_only_some_resources_still_starts_authority_over(arm):
    arm.open()
    arm.submit(reach(joint1=1.0, at=2.0))
    arm.submit(gripper(0.0))
    arm.run(0.3)
    arm.backend.faults.update({("feedback_loss", "arm"), ("feedback_loss", "gripper")})
    arm.run(1.0)  # neither stop is confirmed
    arm.backend.faults.discard(("feedback_loss", "arm"))

    recover = arm.manage("recover")
    assert (recover.state, recover.reason) == ("failed", "recovery_failed")
    assert recover.resources == {"arm": "ready", "gripper": "faulted"}
    # The arm is usable again, but not by the session that held it when it faulted.
    assert recover.new_context_id not in (None, arm.context)
    assert refused(lambda: arm.submit(reach(joint1=0.2))) == "stale_context"


def test_a_stop_reports_failure_when_standstill_is_not_measured(g1):
    g1.open()
    g1.open_stream(forward=0.3)
    g1.run(0.1)
    g1.backend.faults.add(("feedback_loss", "base"))
    stop = g1.runtime.stop(CLIENT, Authority(session_id=g1.session, context_id=g1.context))
    g1.run(1.5)
    stop = g1.runtime.operation(stop.operation_id)
    assert (stop.state, stop.reason) == ("failed", "stop_unconfirmed")
    assert stop.resources == {"base": "faulted", "left_hand": "ready"}
    assert refused(lambda: g1.open_stream(forward=0.1)) == "resource_faulted"


# --------------------------------------------------------------------------
# Blocked motion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "robot, resource, action, settings, reason",
    [
        # A walk fails well before its three seconds are up, and also when it is too short for a sustained error.
        ("g1", "base", drive(0.3, duration=3.0), {}, "tracking_error"),
        ("g1", "base", drive(0.3, duration=0.8), {}, "tracking_error"),
        ("arm", "arm", reach(joint1=1.0, at=2.0), {}, "path_tolerance_violated"),
        # Nothing measures a jaw's progress on the way, so the goal's deadline is what ends it.
        ("arm", "gripper", gripper(0.03), {"max_execution_s": 1.0}, "execution_deadline_exceeded"),
    ],
)
def test_blocked_motion_fails_on_measurement_and_never_succeeds_on_elapsed_time(
    tmp_path, robot, resource, action, settings, reason
):
    with Harness(tmp_path / "journal.sqlite", robot) as harness:
        harness.open()
        harness.backend.faults.add(("blocked", resource))
        goal = harness.finish(harness.submit(action, **settings), 1.5)
        assert (goal.state, goal.result.reason, goal.result.stop_confirmed) == ("failed", reason, True)
        assert harness.effects("stop") == [("stop", (resource,))]

        # The goal failed, not the robot: once it can move again the same action succeeds.
        harness.backend.faults.clear()
        assert harness.finish(harness.submit(action), 4.0).state == "succeeded"


# --------------------------------------------------------------------------
# Restart and reset
# --------------------------------------------------------------------------


def test_a_restart_interrupts_unfinished_goals_and_never_replays_them(arm):
    arm.open()
    old_session, old_context = arm.session, arm.context
    moving = arm.submit(reach(joint1=1.0, at=2.0), request_id="reach")
    arm.run(0.5)
    queued = arm.submit(gripper(0.03), request_id="close-jaw")
    assert (arm.runtime.goal(moving.goal_id).state, queued.state) == ("executing", "accepted")
    last_event = arm.runtime.events(0)[0][-1].sequence
    old_epoch = arm.runtime.observe().samples["arm"].clock_epoch

    arm.boot()  # the process dies and a new incarnation opens the same journal
    assert arm.context != old_context
    # Its measurements are on a clock that started over, and their epoch says so.
    assert arm.runtime.observe().samples["arm"].clock_epoch != old_epoch
    # The old session is over, also for a retry of the request that opened it.
    reopened = arm.runtime.open_session(CLIENT, SessionOpen(request_id="open-1", context_id=old_context))
    assert (reopened.session_id, reopened.state, reopened.reason) == (old_session, "revoked", "runtime_restart")
    assert not arm.runtime.describe().session.held
    # An event reader's place in the old incarnation is answered with a gap, not with silence.
    assert arm.runtime.events(last_event) == ([], True)
    moving = arm.runtime.goal(moving.goal_id)
    assert (moving.state, moving.result.reason) == ("interrupted", "runtime_restart")
    assert moving.result.measurements["dispatch"] == "uncertain"
    assert arm.runtime.goal(queued.goal_id).result.measurements["dispatch"] == "not_dispatched"

    # A retry of the original request finds that operation; it is not executed in the new incarnation.
    retry = arm.submit(reach(joint1=1.0, at=2.0), request_id="reach", session_id=old_session, context_id=old_context)
    assert (retry.goal_id, retry.state) == (moving.goal_id, "interrupted")
    # New work with the old authority is refused.
    stale = lambda: arm.submit(gripper(0.05), session_id=old_session, context_id=old_context)  # noqa: E731
    assert refused(stale) == "stale_context"
    # Sent again in a new session, the request is a different one: it is refused, and the refusal names
    # the operation that the ID belongs to, which is how its outcome is found once the old session is gone.
    arm.open()
    with pytest.raises(RobotError) as conflict:
        arm.submit(reach(joint1=1.0, at=2.0), request_id="reach")
    assert (conflict.value.code, conflict.value.details["operation_id"]) == ("request_conflict", moving.goal_id)
    arm.run(0.5)
    assert arm.effects("apply") == []


@pytest.mark.parametrize(
    "ticked, applied, said",
    [
        (0.1, 0, "targets were dispatched"),
        (STEP, None, "a target may have been dispatched"),
        (0.0, None, "no target had been dispatched"),
    ],
)
def test_a_restart_reports_whether_an_interrupted_stream_had_commanded_motion(g1, ticked, applied, said):
    g1.open()
    stream = g1.open_stream(forward=0.3)
    g1.run(ticked)  # with no tick, the process dies before the controller sees the target
    g1.boot()
    stream = g1.runtime.stream(stream.stream_id)
    # The first dispatch is recorded before the send, and the acknowledgement is recorded separately.
    assert (stream.state, stream.closure, stream.applied_sequence) == ("closed", "interrupted", applied)
    assert said in stream.detail


def test_dropping_old_records_from_memory_never_breaks_a_barrier(tmp_path):
    """Records beyond the retention limit leave memory. That must not cut short a reset or a session's end."""
    with Harness(tmp_path / "journal.sqlite", "arm", limits=ServiceLimits(operation_retention=2)) as harness:
        harness.open()
        earlier = [harness.finish(harness.submit(gripper(width)), 1.0) for width in (0.02, 0.03, 0.04)]
        # The first of them is read from the journal.
        assert [harness.runtime.goal(goal.goal_id).state for goal in earlier] == ["succeeded"] * 3

        moving = harness.submit(reach(joint1=1.0, at=2.0))
        harness.run(0.5)
        waiting = harness.submit(gripper(0.05))
        reset = harness.manage("world.reset", seconds=0.1)
        assert reset.state == "succeeded" and reset.new_context_id != harness.context
        assert [harness.runtime.goal(goal.goal_id).state for goal in (moving, waiting)] == [
            "interrupted",
            "interrupted",
        ]

        harness.context = reset.new_context_id
        harness.open()
        undispatched = [harness.submit(reach(joint1=0.5)), harness.submit(gripper(0.02))]
        applied = len(harness.effects("apply"))
        assert harness.runtime.close_session(CLIENT, harness.session).state == "closed"
        harness.run(0.5)
        assert [harness.runtime.goal(goal.goal_id).state for goal in undispatched] == ["cancelled", "cancelled"]
        assert len(harness.effects("apply")) == applied


def test_a_journal_has_one_owner_at_a_time(tmp_path):
    """Two runtimes on one journal would each answer retries with the other's operations."""
    path = tmp_path / "journal.sqlite"
    assert not held(path)
    with Journal(path, "boot-1"):
        assert held(path)  # how `homunculus demo stop` knows not to remove a live service's files
        with pytest.raises(JournalInUse):
            Journal(path, "boot-2")
    assert not held(path)  # closed, as when its process exits or is killed
    Journal(path, "boot-2").close()


def test_a_full_journal_refuses_new_work_but_still_answers_retries_and_stops(tmp_path):
    """Room is made only from retired contexts. Dropping a current identity would let a retry move the robot twice."""
    with Harness(
        tmp_path / "journal.sqlite", "arm", limits=ServiceLimits(journal_quota=4, operation_retention=2)
    ) as harness:
        old_session = harness.open()
        old_context = harness.context
        first = harness.finish(harness.submit(gripper(0.03), request_id="first"), 1.0)
        for width in (0.05, 0.02):
            harness.finish(harness.submit(gripper(width)), 1.0)
        applied = len(harness.effects("apply"))

        assert refused(lambda: harness.submit(gripper(0.04))) == "journal_full"
        again = harness.submit(gripper(0.03), request_id="first")
        assert (again.goal_id, again.state) == (first.goal_id, "succeeded")
        assert len(harness.effects("apply")) == applied
        # A refused request leaves nothing behind: no operation that never finishes, to crowd the record of
        # the stop below out of the two that memory keeps here.
        for attempt in ("reset-1", "reset-2", "reset-3"):
            reset = ManagementRequest(request_id=attempt, context_id=old_context, operation="world.reset")
            assert refused(lambda reset=reset: harness.runtime.manage(OPERATOR, reset)) == "journal_full"
        stop = harness.runtime.stop(CLIENT, Authority(session_id=old_session, context_id=old_context))
        harness.run(1.0)
        assert harness.runtime.operation(stop.operation_id).state == "succeeded"
        assert harness.effects("world.reset") == []

        harness.boot()  # a restart retires that context, so its finished requests make room
        harness.open()
        assert harness.finish(harness.submit(gripper(0.04)), 1.0).state == "succeeded"
        retry = {"request_id": "first", "session_id": old_session, "context_id": old_context}
        assert refused(lambda: harness.submit(gripper(0.03), **retry)) == "stale_context"


def test_a_world_reset_interrupts_work_and_starts_a_new_context(arm):
    arm.open()
    before = arm.runtime.observe()
    goal = arm.submit(reach(joint1=1.0, at=2.0))
    arm.run(0.5)

    # Two resets are requested before the controller's next tick.
    first, second = (
        arm.runtime.manage(
            OPERATOR, ManagementRequest(request_id=name, context_id=arm.context, operation="world.reset")
        )
        for name in ("reset-1", "reset-2")
    )
    arm.run(0.1)
    reset, late = arm.runtime.operation(first.operation_id), arm.runtime.operation(second.operation_id)
    assert reset.state == "succeeded" and reset.new_context_id != arm.context
    # The second was addressed to the context the first retired, so it does not reset the new world.
    assert (late.state, late.reason) == ("failed", "stale_context")
    assert arm.effects("world.reset") == [("world.reset",)]
    goal = arm.runtime.goal(goal.goal_id)
    assert (goal.state, goal.result.reason) == ("interrupted", "world.reset")
    after = arm.runtime.observe().samples["arm"]
    assert after.values["positions"]["joint1"] == 0.0
    assert after.clock_epoch != before.samples["arm"].clock_epoch  # the new world's clock started over

    assert refused(lambda: arm.submit(gripper(0.03))) == "stale_context"
    old_session, arm.context = arm.session, reset.new_context_id
    assert refused(lambda: arm.submit(gripper(0.03), session_id=old_session)) == "session_invalid"
    arm.open()
    # An observation of the previous world is not evidence about this one.
    conditions = StartConditions(observation_id=before.observation_id, max_position_change=1.0)
    assert refused(lambda: arm.submit(gripper(0.03), start_conditions=conditions)) == "start_condition_failed"
    assert arm.finish(arm.submit(gripper(0.03)), 1.0).state == "succeeded"


# --------------------------------------------------------------------------
# Stopping while the service is busy
# --------------------------------------------------------------------------


def test_requests_that_end_motion_are_answered_while_every_shared_worker_is_busy(arm, monkeypatch):
    """An observation keeps a worker for as long as its cameras take. A stop must not wait for one."""
    tokens = {"client-token": CLIENT, "operator-token": OPERATOR}
    service = Service({"fake-arm": arm.runtime}, tokens, "boot-00000001", ServiceLimits())
    client, operator = ({"Authorization": f"Bearer {token}"} for token in tokens)
    capturing, finish = threading.Semaphore(0), threading.Event()
    observe = arm.runtime.observe

    def slow_observe(cameras=(), after_sequence=None):  # stands in for a camera that has not answered yet
        capturing.release()
        finish.wait(30)
        return observe(cameras, after_sequence)

    arm.open()
    moving = arm.submit(reach(joint1=1.0, at=2.0))
    arm.run(0.4)
    monkeypatch.setattr(arm.runtime, "observe", slow_observe)
    robot = "/v1/robots/fake-arm"

    async def scenario():
        # Leave the pool that plain handlers share two workers, then occupy both with observations.
        anyio.to_thread.current_default_thread_limiter().total_tokens = 2
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://service") as http:
            waiting = [asyncio.create_task(http.get(f"{robot}/observations", headers=client)) for _ in range(2)]
            try:
                for _ in range(2):
                    assert await asyncio.to_thread(capturing.acquire, timeout=10)
                waiting.append(asyncio.create_task(http.get(robot, headers=client)))
                answered, _ = await asyncio.wait(waiting[2:], timeout=0.2)
                assert not answered, "expected the shared workers to be exhausted"

                async def answer(headers, path, body):
                    try:
                        return await asyncio.wait_for(http.post(robot + path, headers=headers, json=body), 10)
                    except TimeoutError:
                        pytest.fail(f"no answer to POST {path} while the observations wait")

                # Accepted or refused, each is answered now, not when a camera lets a worker go.
                gone = {"session_id": "ses-gone", "context_id": arm.context}
                assert (await answer(client, "/stop", gone)).json()["error"]["code"] == "session_invalid"
                authority = {"session_id": arm.session, "context_id": arm.context}
                cancelled = await answer(client, f"/goals/{moving.goal_id}/cancel", authority)
                assert cancelled.json()["state"] == "stopping"
                assert (await answer(operator, "/stop", {})).status_code == 202
            finally:
                finish.set()
                await asyncio.gather(*waiting, return_exceptions=True)

    asyncio.run(scenario())
    arm.run(0.5)
    assert arm.runtime.goal(moving.goal_id).result.stop_confirmed


def test_the_http_stop_worker_is_available_during_historical_journal_reads(arm, monkeypatch):
    service = Service({"fake-arm": arm.runtime}, {"operator-token": OPERATOR}, "boot-00000001", ServiceLimits())
    reads, release = threading.Semaphore(0), threading.Event()
    find = arm.journal.find

    def held_read(*args):
        reads.release()
        assert release.wait(5)
        return find(*args)

    monkeypatch.setattr(arm.journal, "find", held_read)

    async def request_stop():
        transport = httpx.ASGITransport(app=create_app(service))
        headers = {"Authorization": "Bearer operator-token"}
        async with httpx.AsyncClient(transport=transport, base_url="http://service", headers=headers) as http:
            robot = "/v1/robots/fake-arm"
            pending = [asyncio.create_task(http.post(f"{robot}/goals/goal-retired-{i}/cancel")) for i in range(2)]
            try:
                for _ in pending:
                    assert await asyncio.to_thread(reads.acquire, timeout=5)
                stop = await asyncio.wait_for(http.post(f"{robot}/stop"), timeout=2)
                assert stop.status_code == 202
            finally:
                release.set()
                await asyncio.gather(*pending)

    with service:
        asyncio.run(request_stop())


# --------------------------------------------------------------------------
# Two robots, one client
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "robot, action, code",
    [
        ("g1", reach(joint1=0.1), "unsupported_profile"),
        ("g1", posture("thumbs_up"), "unsupported_parameter"),
        ("g1", drive(2.0), "limit_exceeded"),
        # Travel that could not be told apart from standing still: too slow, or over before it can be measured.
        ("g1", drive(0.03), "limit_exceeded"),
        ("g1", drive(0.3, duration=0.4), "limit_exceeded"),
        ("arm", drive(0.2), "unsupported_profile"),
        ("arm", posture("fist"), "unsupported_profile"),
        ("arm", gripper(0.2), "limit_exceeded"),
        ("arm", gripper(0.02, complete_on="contact_detected"), "unsupported_parameter"),
        ("arm", reach(joint1=0.1, at=0.01), "limit_exceeded"),
    ],
)
def test_a_command_the_description_does_not_offer_is_rejected(tmp_path, robot, action, code):
    with Harness(tmp_path / "journal.sqlite", robot) as harness:
        harness.open()
        assert refused(lambda: harness.submit(action)) == code
        harness.run(0.2)
        assert harness.effects("apply") == []


def test_each_description_runs_its_own_profiles_and_declares_what_may_run_together(tmp_path):
    with Harness(tmp_path / "g1.sqlite", "g1") as g1, Harness(tmp_path / "arm.sqlite", "arm") as arm:
        assert {p.profile for p in g1.runtime.describe().profiles} == {"base.planar_velocity/1", "hand.posture/1"}
        assert {p.profile for p in arm.runtime.describe().profiles} == {"joints.trajectory/1", "gripper.width/1"}

        g1.open()
        # Both runtimes share a boot, as the robots of one service do. An observation of the arm is still not
        # one of the base, however many observations the base has issued itself.
        seen_on_arm = arm.runtime.observe().observation_id
        assert g1.runtime.observe().observation_id != seen_on_arm
        assert refused(lambda: g1.open_stream(forward=0.2, basis=seen_on_arm)) == "basis_invalid"
        walk = g1.submit(drive(0.3, duration=1.0))
        # No base-and-hand combination is validated on this description.
        assert refused(lambda: g1.submit(posture("fist"))) == "resource_busy"
        walk = g1.finish(walk, 1.5)
        assert walk.state == "succeeded"
        assert walk.result.measurements["displacement"]["x_m"] == pytest.approx(0.3)
        assert g1.finish(g1.submit(posture("fist")), 1.0).state == "succeeded"


# --------------------------------------------------------------------------
# Backends of your own, and the judges every backend reports
# --------------------------------------------------------------------------


def trajectory(target: str, joints: list[str], positions: list[float], at: float) -> JointTrajectoryGoal:
    point = TrajectoryPoint(time_s=at, positions=positions, velocities=[0.0] * len(positions))
    return JointTrajectoryGoal(profile="joints.trajectory/1", target=target, joints=joints, points=[point])


def test_a_time_between_waypoints_is_sampled_on_the_segment_it_falls_in():
    """The reference that a trajectory judge compares the measured joints against."""
    waypoints = ((0.0, 0.0), (1.0, 1.0), (3.0, 2.0), (4.0, 5.0))
    points = [TrajectoryPoint(time_s=time_s, positions=[q], velocities=[0.0]) for time_s, q in waypoints]
    # A segment that starts and ends at rest is halfway between its ends at half its time.
    for time_s, halfway in ((0.5, 0.5), (2.0, 1.5), (3.5, 3.5)):
        assert sample_trajectory(points, time_s)[0] == pytest.approx([halfway])
    assert sample_trajectory(points, 3.0) == ([2.0], [0.0])  # on a waypoint: that waypoint
    assert sample_trajectory(points, 9.0) == ([5.0], [0.0])  # after the last waypoint: the last, at rest


def test_a_backend_of_your_own_loads_by_import_path_and_is_judged_on_its_measurements(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    robots = load_backends("lift:robots", {"gain": "15"})
    assert list(robots) == ["lift"] and robots["lift"].gain == 15.0
    lift = sys.modules["lift"]

    with Harness(tmp_path / "lift.sqlite", "lift", backend=lambda clock: lift.Lift(clock, gain=15.0)) as harness:
        harness.open()
        goal = harness.finish(harness.submit(trajectory("lift", ["lift"], [0.5], at=2.0)), 3.0)
        assert goal.state == "succeeded"
        assert goal.result.measurements["final_error"]["lift"] <= lift.LIMITS.goal_tolerance

        # A motor too weak to keep up lags the path, and the same judge fails it on what was measured.
        with Harness(tmp_path / "weak.sqlite", "lift", backend=lambda clock: lift.Lift(clock, gain=1.0)) as weak:
            weak.open()
            goal = weak.finish(weak.submit(trajectory("lift", ["lift"], [0.5], at=2.0)), 3.0)
            assert (goal.state, goal.result.reason) == ("failed", "path_tolerance_violated")


def test_a_backend_whose_samples_lack_positions_or_velocities_is_refused_at_start(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import lift

    class Unmeasured(lift.Lift):
        def advance(self):
            feedback = super().advance()
            del feedback.samples["lift"].values["velocities"]
            return feedback

    with pytest.raises(StartupError, match="no positions and velocities maps for resource 'lift'"):
        Harness(tmp_path / "unmeasured.sqlite", "lift", backend=lambda clock: Unmeasured(clock, gain=15.0))


def test_a_service_that_cannot_start_closes_the_backend_it_built(tmp_path, monkeypatch):
    """The backend is built, and then its runtime refuses it. Nothing it holds may outlive that."""
    (tmp_path / "unmeasured.py").write_text(
        "import pathlib\n"
        "import lift\n"
        "from homunculus.runtime import SystemClock\n\n\n"
        "class Unmeasured(lift.Lift):\n"
        "    def advance(self):\n"
        "        feedback = super().advance()\n"
        "        del feedback.samples['lift'].values['velocities']\n"
        "        return feedback\n\n"
        "    def close(self):\n"
        "        pathlib.Path(__file__).with_name('closed').touch()\n\n\n"
        "def robots():\n"
        "    return {'lift': Unmeasured(SystemClock(), gain=15.0)}\n"
    )
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(signal, "signal", lambda *_: None)  # serve() takes over SIGTERM; this process keeps its own
    home = tmp_path / "home"
    with pytest.raises(StartupError, match="no positions and velocities maps for resource 'lift'"):
        serve("unmeasured:robots", {}, home)
    assert (tmp_path / "closed").exists()
    assert not (home / "service.json").exists() and not held(home / "journal.sqlite")


def test_demo_cleanup_holds_off_a_new_service_claim(tmp_path, monkeypatch):
    home = tmp_path / "home"
    demo = home / "demo"
    demo.mkdir(parents=True)
    old_file = demo / "server.log"
    old_file.touch()
    attempted = threading.Event()
    acquired = threading.Event()
    monkeypatch.setenv("HOMUNCULUS_HOME", str(home))
    monkeypatch.setattr(signal, "signal", lambda *_: None)

    class Claimed(Exception):
        pass

    def journal_claim(*args, **kwargs):
        assert not old_file.exists()
        acquired.set()
        raise Claimed

    monkeypatch.setattr(server, "Journal", journal_claim)

    def contender():
        attempted.set()
        with pytest.raises(Claimed):
            serve("fake", {}, demo)

    with ThreadPoolExecutor() as worker:
        claim: Future[None] | None = None

        def check_unowned(path):
            nonlocal claim
            assert not held(path)
            claim = worker.submit(contender)
            assert attempted.wait(5)
            assert not acquired.is_set()
            return False

        monkeypatch.setattr(journal, "held", check_unowned)
        assert cli_main(["demo", "stop"]) == 0
        assert claim is not None
        claim.result(timeout=5)
    assert acquired.is_set()


def task_state(*, finished=False):
    hand = {
        "position": [0.0, 0.0, 0.8],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "gap_m": 0.05,
        "requested_opening": 0.5,
        "touch_normal_force_N": 0.0,
        "arm_joint_angles": [0.0] * 7,
        "controller_status": "holding",
        "finger_joint_angles": {"thumb_0": 0.0},
    }
    return {
        "robot": {
            "hands": {"left": hand, "right": hand},
            "base_xy_yaw": [0.0, 0.0, 0.0],
            "support": {
                "stable": True,
                "linear_speed_m_s": 0.0,
                "angular_speed_rad_s": 0.0,
                "foot_normal_force_N": 1.0,
            },
            "pelvis_height": 0.8,
            "fallen": False,
        },
        "sim_time": 0.0,
        "finished": finished,
    }


@pytest.mark.parametrize("wait_in_queue", [False, True])
def test_a_pending_episode_finish_leaves_stops_responsive(tmp_path, monkeypatch, wait_in_queue):
    entered, release, sent = threading.Event(), threading.Event(), threading.Event()

    def simulator(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True})
        if request.method == "POST" and json.loads(request.content)["action"] == "finish":
            sent.set()
            entered.set()
            assert release.wait(5)
            # The simulator sealed the episode but its reply was lost.
            raise httpx.ReadTimeout("finish response was lost", request=request)
        return httpx.Response(200, json=task_state(finished=sent.is_set()))

    transport = httpx.MockTransport(simulator)
    client = httpx.Client(transport=transport, base_url="http://simulator")
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    with Harness(
        tmp_path / "journal.sqlite", "task", backend=lambda _: TaskBackend("test", "http://simulator", epoch="test")
    ) as task:
        if wait_in_queue:

            def hold_queue():
                entered.set()
                assert release.wait(5)

            task.backend._commands.put(hold_queue)
        command = ManagementRequest(request_id="finish", context_id=task.context, operation="episode.finish")
        finish = task.runtime.manage(OPERATOR, command)
        try:
            task.run(STEP)
            assert entered.wait(5)
            stop = task.runtime.stop(OPERATOR, Authority())
            task.clock.t += 61
            task.run(STEP)
            assert task.runtime.operation(stop.operation_id).operation_id == stop.operation_id
            pending = task.runtime.operation(finish.operation_id)
            assert pending.state == ("failed" if wait_in_queue else "running")
        finally:
            release.set()
        drained = threading.Event()
        task.backend._commands.put(drained.set)
        assert drained.wait(5)
        # Any confirmation read that was queued after the lost reply is ahead of this second barrier.
        drained.clear()
        task.backend._commands.put(drained.set)
        assert drained.wait(5)
        task.run(STEP)
        assert sent.is_set() is not wait_in_queue
        final = task.runtime.operation(finish.operation_id)
        if wait_in_queue:
            assert (final.state, final.reason) == ("failed", "backend_error")
        else:
            assert final.state == "succeeded" and final.outcome["finished"]


def test_a_both_hands_goal_carries_the_right_hand_rigidly_with_the_left(tmp_path, monkeypatch):
    state = task_state()
    hands = state["robot"]["hands"]
    hands["left"] = {**hands["left"], "position": [0.3, 0.2, 0.9]}
    hands["right"] = {**hands["right"], "position": [0.3, -0.2, 0.9]}
    steps = []

    def simulator(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True})
        if request.method == "POST":
            body = json.loads(request.content)
            steps.append(sorted(body.get("hands", {})))
            for side, target in body.get("hands", {}).items():  # arms that track their targets exactly
                hands[side] = {**hands[side], "position": target["position"], "quaternion_wxyz": target["quaternion"]}
            state["sim_time"] += body.get("seconds", 0.0)
        return httpx.Response(200, json=state)

    client = httpx.Client(transport=httpx.MockTransport(simulator), base_url="http://simulator")
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    quarter_turn = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]  # 90 degrees about +z, xyzw
    with Harness(
        tmp_path / "journal.sqlite", "task", backend=lambda _: TaskBackend("test", "http://simulator", epoch="test")
    ) as task:
        task.open()
        action = ToolPoseGoal(
            profile="tool.pose.linear/1",
            target="both_hands",
            position_m=[0.4, 0.2, 1.0],
            orientation_xyzw=quarter_turn,
            speed_m_s=0.1,
        )
        goal = task.submit(action)
        deadline = time.monotonic() + 10
        while (goal := task.runtime.goal(goal.goal_id)).state not in ("succeeded", "failed", "cancelled"):
            assert time.monotonic() < deadline
            task.run(STEP)
            time.sleep(0.002)
    assert goal.state == "succeeded", goal
    assert steps and all(sides == ["left", "right"] for sides in steps if sides)
    # The right hand was 0.4 m to the left hand's -y; a quarter turn about +z puts it 0.4 m to its +x.
    assert hands["left"]["position"] == pytest.approx([0.4, 0.2, 1.0], abs=1e-9)
    assert hands["right"]["position"] == pytest.approx([0.8, 0.2, 1.0], abs=1e-9)
    w, x, y, z = hands["right"]["quaternion_wxyz"]
    assert [x, y, z, w] == pytest.approx(quarter_turn, abs=1e-9)


def test_a_base_goal_with_hands_following_carries_both_hand_targets_with_the_base(tmp_path, monkeypatch):
    state = task_state()
    hands, robot = state["robot"]["hands"], state["robot"]
    hands["left"] = {**hands["left"], "position": [0.3, 0.2, 0.9]}
    hands["right"] = {**hands["right"], "position": [0.3, -0.2, 0.9]}
    carried = []

    def simulator(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True})
        if request.method == "POST":
            body = json.loads(request.content)
            dt = body.get("seconds", 0.0)
            forward, left, turn = body.get("base_velocity", [0.0, 0.0, 0.0])
            x, y, yaw = robot["base_xy_yaw"]  # a base that follows its command exactly
            yaw += turn * dt
            robot["base_xy_yaw"] = [
                x + (math.cos(yaw) * forward - math.sin(yaw) * left) * dt,
                y + (math.sin(yaw) * forward + math.cos(yaw) * left) * dt,
                yaw,
            ]
            carried.append(sorted(body.get("hands", {})))
            for side, target in body.get("hands", {}).items():  # arms that track their targets exactly
                hands[side] = {**hands[side], "position": target["position"], "quaternion_wxyz": target["quaternion"]}
            state["sim_time"] += dt
        return httpx.Response(200, json=state)

    client = httpx.Client(transport=httpx.MockTransport(simulator), base_url="http://simulator")
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    backend = lambda _: TaskBackend("test", "http://simulator", epoch="test", hands_follow_base=True)  # noqa: E731
    with Harness(tmp_path / "journal.sqlite", "task", backend=backend) as task:
        task.open()
        (binding,) = [b for b in task.backend.describe().bindings if b.profile == "base.planar_velocity/1"]
        assert binding.resources == ["base", "left_arm", "right_arm"]
        action = BaseVelocityGoal(
            profile="base.planar_velocity/1",
            target="base",
            forward_m_s=0.2,
            left_m_s=0.0,
            yaw_rate_rad_s=math.pi / 8,
            duration_s=4.0,
        )
        goal = task.submit(action)
        deadline = time.monotonic() + 10
        while (goal := task.runtime.goal(goal.goal_id)).state not in ("succeeded", "failed", "cancelled"):
            assert time.monotonic() < deadline
            task.run(STEP)
            time.sleep(0.002)
    assert goal.state == "succeeded", goal
    assert carried and all(sides == ["left", "right"] for sides in carried)
    # The base turned a quarter turn; each hand kept its place and heading relative to it.
    x, y, yaw = robot["base_xy_yaw"]
    assert yaw == pytest.approx(math.pi / 2, abs=1e-9)
    for side, (ahead, beside) in (("left", (0.3, 0.2)), ("right", (0.3, -0.2))):
        position = hands[side]["position"]
        assert position == pytest.approx([x - beside, y + ahead, 0.9], abs=1e-9)
        w, qx, qy, qz = hands[side]["quaternion_wxyz"]
        assert [qx, qy, qz, w] == pytest.approx([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)], abs=1e-9)


def test_a_task_option_out_of_range_is_refused_before_anything_starts():
    with pytest.raises(StartupError, match="hands_follow_base is true or false"):
        task_robots(url="http://simulator", name="test", hands_follow_base="yes")
    with pytest.raises(StartupError, match="arm_still_m is a distance"):
        task_robots(url="http://simulator", name="test", arm_still_m="0.5")


def test_a_simulator_entry_point_imports_its_neighbours_whatever_path_it_names(tmp_path):
    root = tmp_path / "task"
    entry = root / "environment" / "simulator" / "start.py"
    entry.parent.mkdir(parents=True)
    # As in a task container whose simulator lives in /opt/simulator: the entry point names that path, not its own.
    entry.write_text("import sys\nsys.path.insert(0, '/opt/simulator')\nimport neighbour\n")
    (entry.parent / "neighbour.py").write_text(
        "import os, pathlib, time\npathlib.Path(os.environ['STATE_DIR'], 'imported').touch()\ntime.sleep(60)\n"
    )
    with _start_simulator(root, Path(sys.executable), tmp_path / "episode") as (process, _):
        deadline = time.monotonic() + 10
        while not (tmp_path / "episode" / "imported").exists():
            assert process.poll() is None, (tmp_path / "episode" / "simulator.log").read_text()
            assert time.monotonic() < deadline
            time.sleep(0.05)


def test_jaws_stopped_on_an_object_keep_squeezing_when_the_goal_ends(tmp_path, monkeypatch):
    state = task_state()
    hands = state["robot"]["hands"]
    hands["left"] = {**hands["left"], "gap_m": 0.1, "requested_opening": 1.0}
    openings = []

    def simulator(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True})
        if request.method == "POST":
            body = json.loads(request.content)
            if body["action"] == "grip":  # the jaws close onto a 5 cm object and stop there
                hands["left"] = {**hands["left"], "gap_m": 0.05, "requested_opening": body["opening"]}
            if "openings" in body:
                openings.append(body["openings"]["left"])
                hands["left"] = {**hands["left"], "requested_opening": body["openings"]["left"]}
            state["sim_time"] += body.get("seconds", 0.0)
        return httpx.Response(200, json=state)

    client = httpx.Client(transport=httpx.MockTransport(simulator), base_url="http://simulator")
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    with Harness(
        tmp_path / "journal.sqlite", "task", backend=lambda _: TaskBackend("test", "http://simulator", epoch="test")
    ) as task:
        task.open()
        goal = task.submit(GripperWidthGoal(profile="gripper.width/1", target="left_gripper", width_m=0.0))
        deadline = time.monotonic() + 10
        while (goal := task.runtime.goal(goal.goal_id)).state not in ("succeeded", "failed", "cancelled"):
            assert time.monotonic() < deadline
            task.run(STEP)
            time.sleep(0.002)
        drained = threading.Event()
        task.backend._commands.put(drained.set)
        assert drained.wait(5)
    assert goal.state == "failed" and goal.result.reason == "width_not_reached"
    # The stop that ends the failed goal holds the jaws at the closed request, not at the 5 cm they measure.
    assert openings and all(opening == 0.0 for opening in openings)


@pytest.mark.parametrize(
    ("stops_at_m", "state", "reason"),
    [(0.05, "succeeded", "completed"), (0.0, "failed", "nothing_grasped")],
    ids=["on-an-object", "on-nothing"],
)
def test_a_grasp_succeeds_only_when_an_object_stops_the_jaws_and_keeps_them_closed(
    tmp_path, monkeypatch, stops_at_m, state, reason
):
    state_doc = task_state()
    hands = state_doc["robot"]["hands"]
    hands["left"] = {**hands["left"], "gap_m": 0.1, "requested_opening": 1.0}
    openings = []

    def simulator(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True})
        if request.method == "POST":
            body = json.loads(request.content)
            if body["action"] == "grip":
                hands["left"] = {**hands["left"], "gap_m": stops_at_m, "requested_opening": body["opening"]}
            if "openings" in body:
                openings.append(body["openings"]["left"])
            state_doc["sim_time"] += body.get("seconds", 0.0)
        return httpx.Response(200, json=state_doc)

    client = httpx.Client(transport=httpx.MockTransport(simulator), base_url="http://simulator")
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    with Harness(
        tmp_path / "journal.sqlite", "task", backend=lambda _: TaskBackend("test", "http://simulator", epoch="test")
    ) as task:
        task.open()
        inverted = GripperGraspGoal(
            profile="gripper.grasp/1", target="left_gripper", min_width_m=0.06, max_width_m=0.05
        )
        assert refused(lambda: task.submit(inverted)) == "limit_exceeded"
        goal = task.submit(GripperGraspGoal(profile="gripper.grasp/1", target="left_gripper"))
        deadline = time.monotonic() + 10
        while (goal := task.runtime.goal(goal.goal_id)).state not in ("succeeded", "failed", "cancelled"):
            assert time.monotonic() < deadline
            task.run(STEP)
            time.sleep(0.002)
        drained = threading.Event()
        task.backend._commands.put(drained.set)
        assert drained.wait(5)
    assert (goal.state, goal.result.reason) == (state, reason)
    assert goal.result.measurements["width_m"] == stops_at_m
    # Whatever the verdict, the jaws stay asked to close: nothing they hold is let go.
    assert hands["left"]["requested_opening"] == 0.0 and all(opening == 0.0 for opening in openings)


@pytest.mark.parametrize("launch_child", [False, True])
def test_failed_simulator_start_releases_every_open_resource(tmp_path, monkeypatch, launch_child):
    spawned = []

    @contextmanager
    def simulator():
        root = tmp_path / "task"
        entry = root / "environment" / "simulator" / "start.py"
        entry.parent.mkdir(parents=True)
        entry.write_text("import time\ntime.sleep(60)\n")
        with _start_simulator(root, Path(sys.executable), tmp_path / "episode") as started:
            process, _ = started
            spawned.append(process)
            yield started

    def reply(request):
        document = {"ready": True} if request.url.path == "/health" else {"robot": {}, "sim_time": 0.0}
        return httpx.Response(200, json=document)

    client = httpx.Client(transport=httpx.MockTransport(reply), base_url="http://simulator")
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    with pytest.raises(StartupError, match="does not report the robot state"):
        TaskBackend("test", "http://simulator", epoch="test", simulator=simulator() if launch_child else None)
    assert client.is_closed
    if launch_child:
        (process,) = spawned
        assert process.returncode is not None


def test_a_session_held_for_a_block_is_released_however_the_block_ends(tmp_path, monkeypatch):
    with Harness(tmp_path / "journal.sqlite", "arm") as harness:
        service = Service({"fake-arm": harness.runtime}, {"client-token": CLIENT}, "boot-00000001", ServiceLimits())
        monkeypatch.setenv("HOMUNCULUS_HOME", str(tmp_path / "home"))
        with serving(service) as url:
            store = TargetStore()
            store.connect(url, "client-token")
            with RobotClient(store.get("fake:arm")) as robot:
                with robot.session() as session:
                    assert robot.describe().session.held and session.state == "active"
                assert not robot.describe().session.held

                with pytest.raises(RuntimeError, match="in the block"), robot.session():
                    raise RuntimeError("in the block")
                assert not robot.describe().session.held
                with robot.session() as original:
                    robot.close_session()
                    replacement = robot.open_session()
                assert original.session_id != replacement.session_id
                assert robot.describe().session.held
                assert robot.target.session_id == replacement.session_id
                robot.close_session()

            def fail_save():
                raise OSError("failed to save the session locally")

            with RobotClient(store.get("fake:arm"), on_change=fail_save) as robot:
                with pytest.raises(OSError, match="failed to save the session locally"), robot.session():
                    pytest.fail("the body must not run if saving the acquired session failed")
                assert not robot.describe().session.held


@pytest.mark.parametrize(
    "backend, options, says",
    [
        ("lift", {}, "MODULE:FUNCTION"),
        ("lift:missing", {}, "no function named 'missing'"),
        ("no_such_module:robots", {}, "cannot load"),
        ("lift:robots", {"gian": "3"}, "its options: gain"),
        ("os:getcwd", {}, "must return a dict"),
    ],
)
def test_a_backend_that_cannot_be_loaded_says_why_before_anything_starts(monkeypatch, backend, options, says):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    with pytest.raises(StartupError, match=says):
        load_backends(backend, options)


# Two joints of different kinds, driven by actuators named apart from their joints. No gravity acts on either.
TWO_JOINTS = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="link">
      <joint name="swing" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0" mass="1"/>
      <body name="carriage" pos="0.3 0 0">
        <joint name="reach" type="slide" axis="1 0 0" range="0 0.2"/>
        <geom type="box" size="0.03 0.03 0.03" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="swing_motor" joint="swing" kp="50" kv="5"/>
    <position name="reach_motor" joint="reach" kp="400" kv="40"/>
  </actuator>
  <keyframe><key name="home" qpos="0 0"/></keyframe>
</mujoco>
"""


def two_joints(folder: Path, model: str = TWO_JOINTS) -> Path:
    """A rig folder of your own: the model, and a rig.toml that names it and the joints to drive."""
    folder.mkdir()
    (folder / "two.xml").write_text(model)
    (folder / "rig.toml").write_text('description = "two joints"\nmodel = "two.xml"\narm = ["swing", "reach"]\n')
    return folder


def test_the_mujoco_backend_drives_a_rig_of_your_own(tmp_path):
    pytest.importorskip("mujoco")
    from homunculus.backends.mujoco import MujocoBackend

    folder = two_joints(tmp_path / "two")
    assert list(load_backends("mujoco", {"rig": str(folder)})) == ["two"]  # named after its folder
    assert list(load_backends("mujoco", {"rig": str(folder), "robot_id": "left"})) == ["left"]

    with Harness(tmp_path / "two.sqlite", "two", backend=lambda clock: MujocoBackend(clock, folder)) as harness:
        description = harness.runtime.describe()
        assert [(b.profile, b.target) for b in description.profiles] == [("joints.trajectory/1", "arm")]
        assert [joint.unit for joint in description.profiles[0].limits.joints] == ["rad", "m"]
        harness.open()
        goal = harness.finish(harness.submit(trajectory("arm", ["swing", "reach"], [0.8, 0.15], at=2.0)), 3.0)
        assert goal.state == "succeeded"
        assert max(goal.result.measurements["final_error"].values()) <= 0.02


@pytest.mark.parametrize(
    "change, says",
    [
        (('<keyframe><key name="home" qpos="0 0"/></keyframe>', ""), "keyframe named 'home'"),
        (
            ('<position name="reach_motor" joint="reach" kp="400" kv="40"/>', '<motor joint="reach"/>'),
            "position actuator",
        ),
        (('range="0 0.2"', ""), "slide joint with a range"),
    ],
)
def test_a_mujoco_model_the_backend_cannot_drive_is_refused_with_the_reason(tmp_path, change, says):
    pytest.importorskip("mujoco")
    folder = two_joints(tmp_path / "two", TWO_JOINTS.replace(*change))
    with pytest.raises(StartupError, match=says):
        load_backends("mujoco", {"rig": str(folder)})


WALKING = BaseVelocityLimits(
    profile="base.planar_velocity/1",
    control_point="pelvis",
    frames=["base_heading"],
    max_forward_m_s=0.35,
    max_left_m_s=0.25,
    max_yaw_rate_rad_s=0.5,
    max_goal_duration_s=10.0,
    transient_s=1.0,
    max_tracking_error_m_s=0.05,
    max_tracking_error_rad_s=0.12,
    max_tracking_error_ratio=0.7,
    max_tracking_error_duration_s=0.75,
    tracking_window_s=0.75,
    stop_speed_m_s=0.08,
    stop_yaw_rate_rad_s=0.25,
    settle_time_s=0.2,
)


def walk(speeds: list[float], step: float = 0.25) -> tuple[VelocityJudge, list]:
    """Feed a judge the poses of a walk at these forward speeds, one step each, as a stepped simulator reports."""
    judge = VelocityJudge(drive(0.3, duration=len(speeds) * step), WALKING, 0.0, (0.0, 0.0, 0.0), len(speeds) * step)
    x, statuses = 0.0, []
    for index, speed in enumerate(speeds, start=1):
        x += speed * step
        statuses.append(judge.update(index * step, (x, 0.0, 0.0), still_for_s=0.0))
        if statuses[-1].phase == "failed":
            break
    return judge, statuses


def test_walking_is_judged_on_velocity_averaged_over_a_gait_cycle_and_succeeds_only_once_at_rest():
    # A gait that sways between 0.1 and 0.5 m/s still averages the commanded 0.3 m/s over each cycle.
    judge, statuses = walk([0.1, 0.5] * 6)
    assert [status.phase for status in statuses] == ["executing"] * 11 + ["finishing"]
    assert judge.update(3.4, (1.8, 0.0, 0.0), still_for_s=0.1).phase == "finishing"  # not yet measured at rest
    done = judge.update(3.5, (1.8, 0.0, 0.0), still_for_s=0.2)
    assert done.phase == "done" and done.measurements["displacement"]["x_m"] == pytest.approx(1.8)

    # A base held in place fails as soon as the error has lasted longer than allowed, well before 3 s.
    _, statuses = walk([0.0] * 12)
    failed = statuses[-1]
    assert (failed.phase, failed.error, len(statuses)) == ("failed", "tracking_error", 7)
    assert failed.measurements["measured"]["forward_m_s"] == 0.0


# --------------------------------------------------------------------------
# The client: what it sends and where, and its saved targets
# --------------------------------------------------------------------------


def test_an_empty_request_id_is_refused_instead_of_becoming_a_new_request():
    """`--request-id "$ID"` with ID unset: sent under a new ID, the repeated command would move the robot again."""
    target = Target(
        name="fake:g1", url="http://127.0.0.1:1", robot_id="fake-g1", token="token", session_id="s", context_id="c"
    )
    with RobotClient(target) as client:  # nothing listens there: both requests are refused before they are sent
        with pytest.raises(ValidationError, match="request_id"):
            client.submit_goal(drive(0.3), request_id="")
        with pytest.raises(ValidationError, match="request_id"):
            client.open_stream(
                "base.planar_velocity/1", "base", velocity(0.3), 0.5, basis_observation_id="obs-1", request_id=""
            )


def test_an_empty_id_is_sent_as_written_instead_of_meaning_the_last_one():
    """`homunculus goal cancel "$ID"` with ID unset must not cancel the goal this target recorded last."""
    target = Target(
        name="fake:arm",
        url="http://service",
        robot_id="fake-arm",
        token="token",
        session_id="s",
        context_id="c",
        goal_id="goal-last",
        stream_id="str-last",
        operation_id="op-last",
    )
    asked = []

    def refuse(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "no such id", "details": {}}})

    with RobotClient(target) as client:
        client._http.close()
        client._http = httpx.Client(transport=httpx.MockTransport(refuse))
        for call in (client.goal, client.cancel_goal, client.stream, client.close_stream, client.operation):
            with pytest.raises(RobotError, match="no such id"):
                call("")
    assert len(asked) == 5 and not any("last" in path for path in asked)


@pytest.mark.parametrize(
    "argv",
    [
        ["goal", "cancel", ""],
        ["goal", "inspect", " "],
        ["stream", "close", ""],
        ["operation", "inspect", ""],
        ["-t", "", "describe"],
        ["describe", "--target", ""],
        ["forget", ""],
        ["goal", "joints", "joint1=1", "--request-id", ""],
        ["stream", "update", "--forward", "0.1", "--basis", ""],
        ["observe", "--camera", ""],
        ["video", ""],
    ],
)
def test_an_empty_id_or_target_name_typed_on_the_command_line_is_a_usage_error(argv, capsys):
    """`homunculus goal cancel "$GOAL"` with GOAL unset is refused before anything is read or sent."""
    with pytest.raises(SystemExit) as exited:
        cli_main(argv)
    assert exited.value.code == 2
    assert "it is empty; if a shell variable should hold it, that variable is not set" in capsys.readouterr().err


def test_an_empty_target_name_is_unknown_instead_of_the_default(tmp_path):
    store = TargetStore(tmp_path)
    store.targets["fake:arm"] = Target(name="fake:arm", url="http://127.0.0.1:1", robot_id="fake-arm", token="t")
    store.default = "fake:arm"
    assert store.get().name == "fake:arm"
    with pytest.raises(ClientError, match="unknown target ''"):
        store.get("")


def test_a_proxy_named_in_the_environment_is_sent_nothing(monkeypatch):
    """HTTP_PROXY is set for other programs. A robot's commands and tokens are not for that proxy, and a
    stop must not depend on it."""
    with socket.socket() as proxy:
        proxy.bind(("127.0.0.1", 0))
        proxy.listen()
        proxy.setblocking(False)
        for name in ("HTTP_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(name, f"http://127.0.0.1:{proxy.getsockname()[1]}")
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        target = Target(
            name="fake:g1", url="http://127.0.0.1:1", robot_id="fake-g1", token="token", operator_token="operator"
        )
        # Nothing listens at the target's address, and that is where the stop is asked and refused.
        with RobotClient(target, timeout_s=0.5, retries=0) as client, pytest.raises(ClientError, match="cannot reach"):
            client.stop(as_operator=True)
        with pytest.raises(BlockingIOError):
            proxy.accept()  # no connection came to the proxy


def test_a_command_saves_only_what_it_changed_in_the_shared_targets_file(tmp_path):
    """Commands overlap. One that started earlier must not undo what another saved in the meantime."""
    store = TargetStore(tmp_path)
    for name in ("fake:arm", "fake:g1"):
        robot_id = name.replace(":", "-")
        store.targets[name] = Target(name=name, url="http://127.0.0.1:1", robot_id=robot_id, token="token")
    store.default = "fake:arm"
    store.save()

    following, working = TargetStore(tmp_path), TargetStore(tmp_path)  # `events --follow`, and another shell
    working.targets["fake:arm"].session_id = "ses-1"
    working.forget(["fake:g1"])
    working.save()
    following.targets["fake:arm"].goal_id = "goal-1"
    following.save()  # its view still has no session, and still has fake:g1

    saved = TargetStore(tmp_path)
    assert (saved.targets["fake:arm"].session_id, saved.targets["fake:arm"].goal_id) == ("ses-1", "goal-1")
    assert (list(saved.targets), saved.default) == (["fake:arm"], "fake:arm")

    # Forgetting the default robot puts no other in its place: the next command has to name the one it means.
    saved.targets["fake:g1"] = Target(name="fake:g1", url="http://127.0.0.1:1", robot_id="fake-g1", token="token")
    saved.forget(["fake:arm"])
    saved.save()
    assert (list(TargetStore(tmp_path).targets), TargetStore(tmp_path).default) == (["fake:g1"], None)


def test_two_commands_that_save_in_the_same_instant_both_keep_what_they_learned(tmp_path, monkeypatch):
    """`homunculus -t fake:arm session open & homunculus -t fake:g1 session open`: neither session ID may be lost."""
    store = TargetStore(tmp_path)
    for name in ("fake:arm", "fake:g1"):
        robot_id = name.replace(":", "-")
        store.targets[name] = Target(name=name, url="http://127.0.0.1:1", robot_id=robot_id, token="token")
    store.save()
    first, second = TargetStore(tmp_path), TargetStore(tmp_path)
    first.targets["fake:arm"].session_id, second.targets["fake:g1"].session_id = "ses-arm", "ses-g1"

    # Hold the first save between reading the file and replacing it, which is where the second arrives.
    has_read, may_write = threading.Event(), threading.Event()
    read = TargetStore._read

    def held_read(saving):
        stored = read(saving)
        if saving is first:
            has_read.set()
            may_write.wait(10)
        return stored

    monkeypatch.setattr(TargetStore, "_read", held_read)
    saves = [threading.Thread(target=first.save), threading.Thread(target=second.save)]
    saves[0].start()
    assert has_read.wait(10)
    saves[1].start()
    saves[1].join(0.2)  # long enough for the second save to finish, if nothing made it wait its turn
    may_write.set()
    for save in saves:
        save.join(10)

    saved = TargetStore(tmp_path).targets
    assert (saved["fake:arm"].session_id, saved["fake:g1"].session_id) == ("ses-arm", "ses-g1")


def test_a_late_save_cannot_attach_the_old_robots_session_to_a_replacement(tmp_path, monkeypatch):
    def robots(*args, **kwargs):
        return httpx.Response(200, json={"robots": [{"robot_id": "fake-arm"}]})

    monkeypatch.setattr(httpx, "get", robots)
    first = TargetStore(tmp_path)
    first.connect("http://old", "example")
    first.save()

    reconnect = TargetStore(tmp_path)
    reconnect.connect("http://new", "example")
    reconnect.save()
    first.targets["fake:arm"].session_id = "ses-old"
    first.targets["fake:arm"].context_id = "ctx-old"
    first.save()

    target = TargetStore(tmp_path).targets["fake:arm"]
    assert (target.url, target.session_id, target.context_id) == ("http://new", None, None)


def test_a_failed_target_save_can_be_retried_on_the_same_store(tmp_path, monkeypatch):
    store = TargetStore(tmp_path)
    target = Target(name="fake:arm", url="http://example", robot_id="fake-arm", token="example")
    store.targets[target.name] = target
    store.save()
    target.session_id = "ses-new"

    def fail(*args):
        raise OSError("injected rename failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="injected rename failure"):
            store.save()
    store.save()
    assert TargetStore(tmp_path).get(target.name).session_id == "ses-new"


# --------------------------------------------------------------------------
# Cameras and live video
# --------------------------------------------------------------------------


class Filmed(FakeBackend):
    """The fake arm with two cameras: `front` can be watched live, `wrist` takes snapshots only. `gallery` is
    for people: it plays live but is not one of the robot's cameras."""

    def __init__(self, clock: Clock):
        super().__init__("arm", clock)
        self.captured = itertools.count(1)

    def describe(self) -> BackendFacts:
        facts = super().describe()
        facts.sensors = [
            *facts.sensors,
            SensorInfo(id="front", kind="camera", provenance="fake", description="numbered frames", video_fps=50.0),
            SensorInfo(id="wrist", kind="camera", provenance="fake", description="each capture spends a budget"),
            SensorInfo(
                id="gallery", kind="spectator_camera", provenance="fake", description="for people", video_fps=50.0
            ),
        ]
        return facts

    def capture(self, sensor: str) -> list[Frame]:
        data = f"{sensor} {next(self.captured)}".encode()
        image = CameraImage(
            kind="rgb",
            media_type="image/png",
            width=1,
            height=1,
            clock_id="fake_execution",
            clock_epoch="test",
            capture_time_ns="0",
            focal_px=[1.0, 1.0],
            principal_px=[0.5, 0.5],
            camera_pose=Pose(frame_id="world", position_m=[0.0, 0.0, 1.0], orientation_xyzw=[1.0, 0.0, 0.0, 0.0]),
        )
        return [Frame(image, data)]


def eventually(condition: Callable[[], bool], within_s: float = 5.0) -> bool:
    """Whether the condition holds within the time: what the service's own threads do takes a moment to show."""
    deadline = time.monotonic() + within_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


@contextmanager
def serving(service: Service) -> Iterator[str]:
    """The service on a free local port, as `homunculus serve` runs it: a video response never ends, so it is read
    from a real connection."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        server = uvicorn.Server(uvicorn.Config(create_app(service), log_level="warning", timeout_graceful_shutdown=1))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            assert eventually(lambda: server.started, within_s=10)
            yield f"http://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            server.should_exit = True  # a thread has no `with`: it is told to end and waited for here
            thread.join(10)


def video_parts(chunks: Iterator[bytes]) -> Iterator[tuple[str, bytes]]:
    """A live video's parts in the order they come, as each one's content type and bytes."""
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while b"\r\n\r\n" in buffer:
            head, rest = buffer.split(b"\r\n\r\n", 1)
            boundary, *lines = head.decode().split("\r\n")
            assert boundary == "--frame"
            headers = dict(line.split(": ", 1) for line in lines)
            length = int(headers["Content-Length"])
            if len(rest) < length + 2:
                break
            yield headers["Content-Type"], rest[:length]
            buffer = rest[length + 2 :]


def test_a_camera_plays_live_through_a_link_that_needs_no_token(tmp_path, monkeypatch, capsys):
    with Harness(tmp_path / "journal.sqlite", "fake-arm", backend=Filmed) as harness:
        limits = ServiceLimits(max_video_viewers=1)
        service = Service({"fake-arm": harness.runtime}, {"client-token": CLIENT}, "boot-00000001", limits)
        with serving(service) as url:
            monkeypatch.setenv("HOMUNCULUS_HOME", str(tmp_path / "home"))
            store = TargetStore()
            store.connect(url, "client-token")
            store.save()
            assert cli_main(["--json", "video", "front"]) == 0
            link = VideoLink.model_validate_json(capsys.readouterr().out)
            assert (link.camera, link.fps, link.expires_in_s) == ("front", 50.0, 600.0)
            assert link.url.startswith(f"{url}/v1/robots/fake-arm/video?camera=front&until=")

            with httpx.stream("GET", link.url, timeout=10) as watching:  # no token: the link lets it in
                assert watching.status_code == 200
                assert watching.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
                parts = video_parts(watching.iter_bytes())  # kept to the end: dropping it closes the connection
                assert [next(parts) for _ in range(3)] == [("image/png", f"front {n}".encode()) for n in (1, 2, 3)]
                with httpx.stream("GET", link.url, timeout=10) as second:
                    assert second.status_code == 503  # the one viewer allowed is watching
                assert next(parts) == ("image/png", b"front 4")  # and goes on watching
            assert eventually(lambda: service.viewers == 0)  # a viewer that leaves gives its place back


def test_observe_saves_the_robots_camera_images_and_says_how_each_was_taken(tmp_path, monkeypatch, capsys):
    """An agent gets a file it can open, not a handle no command takes. A spectator camera is never recorded."""
    with Harness(tmp_path / "journal.sqlite", "fake-arm", backend=Filmed) as harness:
        service = Service({"fake-arm": harness.runtime}, {"client-token": CLIENT}, "boot-00000001", ServiceLimits())
        with serving(service) as url:
            monkeypatch.setenv("HOMUNCULUS_HOME", str(tmp_path / "home"))
            store = TargetStore()
            store.connect(url, "client-token")
            store.save()

            assert cli_main(["--json", "observe", "--camera", "wrist", "--camera", "front"]) == 0
            printed = json.loads(capsys.readouterr().out)
            files = [Path(image.pop("file")) for image in printed["images"]]
            observation = Observation.model_validate(printed)
            assert [image.sensor for image in observation.images] == ["wrist", "front"]
            assert all(path.parent == tmp_path / "home" / "images" for path in files)
            assert [path.read_bytes() for path in files] == [b"wrist 1", b"front 2"]
            assert observation.images[0].camera_pose.position_m == [0.0, 0.0, 1.0]

            assert cli_main(["observe", "--camera", "wrist", "--save", str(tmp_path / "shots")]) == 0
            text = capsys.readouterr().out
            assert f"image wrist rgb 1x1 image/png: {tmp_path / 'shots'}" in text
            assert "lens [1, 1] px focal, centre [0.5, 0.5] px; camera at [0, 0, 1] m" in text

            assert cli_main(["observe", "--camera", "gallery"]) != 0
            assert "'gallery' is for people watching its live video" in capsys.readouterr().err
            assert cli_main(["describe"]) == 0
            described = capsys.readouterr().out
            assert "cameras     front (live video, 50 fps), wrist" in described
            assert "spectators  gallery (live video, 50 fps)" in described


def test_the_task_states_its_camera_as_mujoco_does_once_converted(panda):
    """The task simulator sends a MuJoCo camera's own numbers; converted, they match the MuJoCo backend's."""
    pytest.importorskip("PIL")
    from homunculus.backends.task import _camera_image

    frame = panda.backend.capture("wrist")[0]
    model, shown = panda.backend._model, panda.backend._shown
    camera = model.camera("wrist").id
    task = _camera_image(
        {
            "width": 640,
            "height": 480,
            "focal_px": 480 / (2 * math.tan(math.radians(model.cam_fovy[camera]) / 2)),
            "camera_position": shown.cam_xpos[camera].tolist(),
            "world_from_camera": shown.cam_xmat[camera].reshape(3, 3).tolist(),
            "sim_time": shown.time,
        },
        "state",
    )
    ours = frame.image
    assert task.focal_px == pytest.approx(ours.focal_px) and task.principal_px == ours.principal_px
    assert task.camera_pose.position_m == pytest.approx(ours.camera_pose.position_m)
    same = sum(a * b for a, b in zip(task.camera_pose.orientation_xyzw, ours.camera_pose.orientation_xyzw, strict=True))
    assert abs(same) == pytest.approx(1.0)  # q and -q are the same rotation


def test_a_viewer_who_leaves_before_the_first_frame_gives_its_place_back(tmp_path, monkeypatch):
    """Otherwise each such viewer would hold a place until the service restarts."""
    with Harness(tmp_path / "journal.sqlite", "fake-arm", backend=Filmed) as harness:
        service = Service({"fake-arm": harness.runtime}, {"client-token": CLIENT}, "boot-00000001", ServiceLimits())
        left, capture = threading.Event(), harness.backend.capture

        def held(sensor: str) -> list[Frame]:
            left.wait(10)  # the first frame is still being captured when the viewer leaves
            return capture(sensor)

        monkeypatch.setattr(harness.backend, "capture", held)
        until = int(time.time()) + 60
        query = httpx.QueryParams(
            camera="front", until=until, signature=service.video_signature("fake-arm", "front", until)
        )
        with serving(service) as url:
            with socket.create_connection(("127.0.0.1", int(url.rsplit(":", 1)[1]))) as viewer:
                viewer.sendall(f"GET /v1/robots/fake-arm/video?{query} HTTP/1.1\r\nHost: robot\r\n\r\n".encode())
                assert eventually(lambda: service.viewers == 1)
            time.sleep(0.2)  # the viewer has left; this is for the service to see the connection end
            left.set()
            assert eventually(lambda: service.viewers == 0)


def test_a_video_that_cannot_play_is_refused_with_the_reason(tmp_path, monkeypatch):
    """A browser shows the refusal, so a video that will not play never looks like a blank one."""
    with Harness(tmp_path / "journal.sqlite", "fake-arm", backend=Filmed) as harness:
        service = Service({"fake-arm": harness.runtime}, {"client-token": CLIENT}, "boot-00000001", ServiceLimits())
        robot = "/v1/robots/fake-arm"
        with serving(service) as url, httpx.Client(base_url=url, timeout=10) as http:
            no_token = http.get(f"{robot}/video-link", params={"camera": "front"})
            assert no_token.json()["error"]["code"] == "unauthenticated"
            client = {"Authorization": "Bearer client-token"}
            snapshots = http.get(f"{robot}/video-link", params={"camera": "wrist"}, headers=client).json()["error"]
            assert snapshots["code"] == "unsupported_sensor"
            assert (
                "takes snapshots only" in snapshots["message"]
                and "each capture spends a budget" in snapshots["message"]
            )
            absent = http.get(f"{robot}/video-link", params={"camera": "rear"}, headers=client).json()["error"]
            assert (absent["code"], absent["details"]["cameras"]) == (
                "unsupported_sensor",
                ["front", "gallery", "wrist"],
            )

            def refusal(camera: str, until: int, signature: str) -> dict:
                """Why a link does not play. One that plays fails the test instead of streaming forever."""
                query = {"camera": camera, "until": until, "signature": signature}
                with http.stream("GET", f"{robot}/video", params=query) as response:
                    assert response.status_code != 200, "this link should not play"
                    response.read()
                    return response.json()["error"]

            until = int(time.time()) + 60
            signature = service.video_signature("fake-arm", "front", until)
            assert refusal("wrist", until, signature)["code"] == "unauthenticated"  # made for another camera
            assert refusal("front", until + 600, signature)["code"] == "unauthenticated"  # its time moved on
            earlier = Service({"fake-arm": harness.runtime}, {}, "boot-00000000", ServiceLimits())  # restarted since
            assert (
                refusal("front", until, earlier.video_signature("fake-arm", "front", until))["code"]
                == "unauthenticated"
            )
            past = int(time.time()) - 1
            expired = refusal("front", past, service.video_signature("fake-arm", "front", past))
            assert (expired["code"], "expired" in expired["message"]) == ("unauthenticated", True)
            # Signed or not, a camera that takes snapshots only has no video to play.
            assert refusal("wrist", until, service.video_signature("fake-arm", "wrist", until))["code"] == (
                "unsupported_sensor"
            )

            # A camera that cannot capture is refused before the response begins, and takes no viewer's place.
            def unplugged(sensor: str) -> list[Frame]:
                raise RobotError("backend_error", f"camera {sensor!r} is unplugged")

            monkeypatch.setattr(harness.backend, "capture", unplugged)
            failed = refusal("front", until, signature)
            assert (failed["code"], failed["message"], service.viewers) == (
                "backend_error",
                "camera 'front' is unplugged",
                0,
            )


@pytest.fixture
def panda(tmp_path) -> Iterator[Harness]:
    """The Franka Emika Panda that `--backend mujoco` serves, on the test's clock.

    Its files come from MuJoCo Menagerie: downloaded the first time, then read from Menagerie's cache. Its
    backend closes when the test ends, which frees the renderer on the renderer's own thread.
    """
    pytest.importorskip("mujoco")
    from homunculus.backends.mujoco import MujocoBackend

    with Harness(
        tmp_path / "sim.sqlite", "panda", backend=lambda clock: MujocoBackend(clock, rigs.find("panda"))
    ) as sim:
        yield sim


def test_the_fake_arm_is_the_simulated_panda_without_physics(arm, panda):
    fake, sim = arm, panda

    def ranges(harness: Harness) -> tuple[list[tuple[str, float, float]], float, float]:
        limits = {binding.profile: binding.limits for binding in harness.runtime.describe().profiles}
        arm, gripper = limits["joints.trajectory/1"], limits["gripper.width/1"]
        return (
            [(j.name, j.min_position, j.max_position) for j in arm.joints],
            gripper.min_width_m,
            gripper.max_width_m,
        )

    assert ranges(fake) == ranges(sim)
    start = [harness.runtime.observe().samples["arm"].values["positions"] for harness in (fake, sim)]
    assert start[0] == pytest.approx(PANDA_HOME) and start[1] == pytest.approx(PANDA_HOME, abs=1e-3)


def test_the_simulated_panda_closes_its_gripper_through_its_one_tendon_actuator(panda):
    panda.open()
    goal = panda.finish(panda.submit(gripper(0.02)), 2.0)
    assert goal.state == "succeeded"
    fingers = panda.runtime.observation(goal.result.completion_observation_id).samples["gripper"].values
    assert fingers["width_m"] == pytest.approx(0.02, abs=0.003)
    # One actuator pulls a tendon over both fingers; the model's equality constraint keeps them level.
    assert fingers["positions"]["finger_joint1"] == pytest.approx(fingers["positions"]["finger_joint2"], abs=1e-3)


def test_the_simulated_panda_sees_through_its_own_cameras_and_says_where_each_one_was(panda):
    """The lens and pose each image carries map a point of the world onto the pixel that shows it."""
    image_module = pytest.importorskip("PIL.Image")
    cameras = {sensor.id: sensor.kind for sensor in panda.runtime.describe().sensors if "camera" in sensor.kind}
    assert cameras == {"wrist": "camera", "workspace": "camera", "overview": "spectator_camera"}
    with pytest.raises(RobotError, match="for people watching its live video") as refused:
        panda.runtime.observe(["overview"])
    assert refused.value.details["cameras"] == ["workspace", "wrist"]
    assert panda.runtime.video_fps("overview") == 20.0  # people can still watch it

    # Above the mug, as an agent would put the hand before grasping it.
    panda.open()
    above = reach(
        joint1=0.0979,
        joint2=0.1032,
        joint3=0.1297,
        joint4=-2.2497,
        joint5=-0.0176,
        joint6=2.3383,
        joint7=-2.117,
        at=3.0,
    )
    assert panda.finish(panda.submit(above), 4.0).state == "succeeded"
    observation = panda.runtime.observe(["wrist", "workspace"])
    assert [(i.sensor, i.kind, i.media_type, i.width, i.height) for i in observation.images] == [
        ("wrist", "rgb", "image/jpeg", 640, 480),
        ("workspace", "rgb", "image/jpeg", 640, 480),
    ]
    mug = panda.backend._data.body("mug").xpos.copy()
    for image in observation.images:
        # The mug's origin, seen from the camera, then projected through the lens.
        x, y, z, w = image.camera_pose.orientation_xyzw
        rotation = [  # world from camera
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        offset = [m - c for m, c in zip(mug, image.camera_pose.position_m, strict=True)]
        seen = [sum(rotation[row][axis] * offset[row] for row in range(3)) for axis in range(3)]
        assert seen[2] > 0  # in front of the camera
        column = image.focal_px[0] * seen[0] / seen[2] + image.principal_px[0]
        row = image.focal_px[1] * seen[1] / seen[2] + image.principal_px[1]
        _, jpeg = panda.runtime.image(image.handle)
        red, green, blue = image_module.open(io.BytesIO(jpeg)).convert("RGB").getpixel((round(column), round(row)))
        assert blue > red + 20 and green > red + 20, (
            image.sensor,
            column,
            row,
        )  # the blue-green mug, not the table

    moved = panda.runtime.video_frame("overview")
    assert moved.image.media_type == "image/jpeg" and moved.data.startswith(b"\xff\xd8")
    assert int(moved.image.capture_time_ns) >= int(observation.images[0].capture_time_ns)
    assert moved.image.clock_epoch == observation.images[0].clock_epoch == panda.runtime.describe().world_id


# --------------------------------------------------------------------------
# A profile added from outside homunculus
# --------------------------------------------------------------------------


class DwellGoal(Model):
    """A profile homunculus does not ship: keep a resource as it is for a while."""

    profile: Literal["test.dwell/1"]
    target: str
    seconds: float = Field(gt=0)


class DwellLimits(Model):
    profile: Literal["test.dwell/1"]
    max_seconds: float
    settle_time_s: float = 0.0


class DwellJudge(MotionJudge):
    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    @measures
    def update(self, elapsed_s: float, width_m: float) -> ActivityStatus:
        if elapsed_s < self.seconds:
            return ActivityStatus("executing")
        return ActivityStatus("done", {"dwelled_s": elapsed_s, "width_m": width_m})


def _resolve_dwell(action: DwellGoal, limits: DwellLimits) -> DwellGoal:
    if action.seconds > limits.max_seconds:
        raise RobotError("limit_exceeded", f"seconds={action.seconds:g} exceeds {limits.max_seconds:g}")
    return action


@pytest.fixture
def dwell() -> Iterator[Profile]:
    profile = register(
        Profile(
            name="test.dwell/1",
            action=DwellGoal,
            limits=DwellLimits,
            resolve=_resolve_dwell,
            duration_s=lambda action: action.seconds,
            judge=lambda action, limits, reading: DwellJudge(action.seconds),
            measure=lambda judge, reading: judge.update(reading.elapsed_s, reading.values["width_m"]),
        )
    )
    yield profile
    del PROFILES[profile.name]


class DwellingArm(FakeBackend):
    """The fake arm with test.dwell/1 bound to its gripper; the runtime judges it from the gripper's samples."""

    reports_goal_samples = False

    def __init__(self, clock: Clock) -> None:
        super().__init__("arm", clock)
        self.dwells: dict[str, tuple[float, float]] = {}  # activity id -> (started at, seconds)

    def describe(self) -> BackendFacts:
        facts = super().describe()
        binding = ProfileBinding(
            profile="test.dwell/1",
            target="gripper",
            role="gripper",
            resources=["gripper"],
            modes=["goal"],
            controller="test_dwell",
            assistance="none: nothing moves",
            stop_deadline_wall_s=0.5,
            limits=DwellLimits(profile="test.dwell/1", max_seconds=5.0),
        )
        return dataclasses.replace(facts, bindings=[*facts.bindings, binding])

    def apply(self, activity_id: str, contract) -> None:
        self.effects.append(("apply", activity_id, contract.action.profile, contract.resources[0]))
        self.dwells[activity_id] = (self._time, contract.action.seconds)

    def advance(self):
        feedback = super().advance()
        for activity_id, (started, seconds) in self.dwells.items():
            feedback.activities[activity_id] = ActivityStatus(
                "done" if self._time - started >= seconds else "executing"
            )
        return feedback


def test_a_profile_registered_outside_homunculus_is_admitted_judged_and_reported(tmp_path, dwell):
    with Harness(tmp_path / "journal.sqlite", "arm", backend=DwellingArm) as arm:
        arm.open()
        assert "test.dwell/1" in [b.profile for b in arm.runtime.describe().profiles]
        too_long = DwellGoal(profile="test.dwell/1", target="gripper", seconds=9.0)
        assert refused(lambda: arm.submit(too_long)) == "limit_exceeded"

        goal = arm.submit(DwellGoal(profile="test.dwell/1", target="gripper", seconds=1.0))
        assert arm.finish(goal, 0.5).state == "executing"
        goal = arm.finish(goal, 1.0)
        assert (goal.state, goal.result.reason) == ("succeeded", "completed")
        assert goal.result.measurements["dwelled_s"] >= 1.0
        assert goal.result.measurements["width_m"] == pytest.approx(0.08)  # the runtime handed it the gripper's sample
        assert goal.retained.profile == "test.dwell/1"


def test_a_registered_profile_crosses_the_wire_as_its_own_model(dwell):
    wire = {"profile": "test.dwell/1", "target": "gripper", "seconds": 2.0}
    spec = GoalSpec.model_validate({"action": wire})
    assert isinstance(spec.action, DwellGoal)
    assert GoalSpec.model_validate_json(spec.model_dump_json()) == spec
    with pytest.raises(ValidationError):
        GoalSpec.model_validate({"action": {**wire, "seconds": -1.0}})  # the profile's own model still validates


def test_a_profile_this_process_never_registered_is_kept_as_sent_and_the_robot_refuses_it(arm):
    wire = {"profile": "test.nothing/1", "target": "gripper", "grip": 3}
    spec = GoalSpec.model_validate({"action": wire})
    assert spec.model_dump(mode="json")["action"] == wire  # a client can still show it
    arm.open()
    assert refused(lambda: arm.submit(spec.action)) == "unsupported_profile"


def test_a_backend_binding_a_profile_no_one_registered_does_not_start(tmp_path):
    with pytest.raises(StartupError, match=r"binds test\.dwell/1, which no one registered"):
        Harness(tmp_path / "journal.sqlite", "arm", backend=DwellingArm)


def test_register_refuses_a_name_its_models_do_not_carry_or_one_already_taken(dwell):
    renamed = dataclasses.replace(dwell, name="test.other/1")
    with pytest.raises(ValueError, match="must be Literal"):
        register(renamed)
    with pytest.raises(ValueError, match="already registered"):
        register(dwell)
