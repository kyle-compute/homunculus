"""A deterministic fake robot for protocol checks. It is not physics and not a robot.

Joints follow their commands kinematically on the clock the runtime uses, so a
test that advances a manual clock gets the same result on every run. A result
from this backend is evidence about the protocol only.

Two descriptions with different capabilities are built in:

    g1    a base that takes velocity goals and streams, and one hand
    arm   the Franka Emika Panda's seven arm joints, which take trajectories, and its gripper

``faults`` injects the failures the contract tests need, as (fault, resource)
pairs: ``feedback_loss`` invalidates a resource's measurements, ``blocked``
keeps it from moving, and ``stop_lost`` makes a stop command that includes it
fail undelivered. ``effects`` records every command the backend received.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..models import (
    BackendInfo,
    BaseVelocityLimits,
    BaseVelocityTarget,
    ClockInfo,
    Contract,
    GripperWidthGoal,
    GripperWidthLimits,
    HandPostureGoal,
    HandPostureLimits,
    JointLimit,
    JointTrajectoryGoal,
    JointTrajectoryLimits,
    ProfileBinding,
    RobotError,
    Sample,
    SensorInfo,
)
from ..profiles import (
    ActivityStatus,
    PostureJudge,
    TrajectoryJudge,
    VelocityJudge,
    WidthJudge,
    sample_trajectory,
)
from ..runtime import Backend, BackendFacts, Clock, Feedback, ResourceSpec, SystemClock

NOTICE = "Fake robot: deterministic kinematics with no physics and no robot. Results are protocol evidence only."
PERIOD_S = 0.02
ZERO_VELOCITY = {"forward_m_s": 0.0, "left_m_s": 0.0, "yaw_rate_rad_s": 0.0}
_Judge = TrajectoryJudge | WidthJudge | PostureJudge | VelocityJudge


@dataclass
class _Motion:
    """One goal or stream being executed on a group."""

    activity_id: str
    started: float
    # Holds the goal's action and limits, or a stream's newest setpoint; the controller follows what it holds.
    judge: _Judge
    failure: ActivityStatus | None = None  # once failed, the controller gives up and waits to be stopped


@dataclass
class _Group:
    """A control group's kinematic state and what its controller holds when idle."""

    spec: ResourceSpec
    positions: dict[str, float]
    velocities: dict[str, float]
    hold: dict[str, float]
    # Standstill: every measured speed at or below its threshold for settle_time_s.
    still_speed: dict[str, float]
    settle_time_s: float
    still_for: float = 0.0
    motion: _Motion | None = None
    sample: Sample | None = None


@dataclass
class _Robot:
    frames: list[str]
    concurrent: list[list[str]]
    bindings: list[ProfileBinding]
    groups: list[_Group]


def _joint_group(rid: str, role: str, positions: dict[str, float], still_speed: float, settle_time_s: float) -> _Group:
    return _Group(
        spec=ResourceSpec(id=rid, role=role, joints=list(positions), controller=f"fake_{rid}_position"),
        positions=dict(positions),
        velocities=dict.fromkeys(positions, 0.0),
        hold=dict(positions),
        still_speed=dict.fromkeys(positions, still_speed),
        settle_time_s=settle_time_s,
    )


def _g1() -> _Robot:
    base = BaseVelocityLimits(
        profile="base.planar_velocity/1",
        control_point="base_origin",
        frames=["base_heading"],
        max_forward_m_s=0.6,
        max_left_m_s=0.3,
        max_yaw_rate_rad_s=0.8,
        max_goal_duration_s=10.0,
        min_update_validity_wall_s=0.05,
        max_update_validity_wall_s=1.0,
        transient_s=0.5,
        max_tracking_error_m_s=0.05,
        max_tracking_error_rad_s=0.1,
        max_tracking_error_duration_s=0.5,
        stop_speed_m_s=0.02,
        stop_yaw_rate_rad_s=0.05,
        settle_time_s=0.2,
    )
    fingers = ["left_thumb", "left_index", "left_middle"]
    hand = HandPostureLimits(
        profile="hand.posture/1",
        postures={
            "open": dict.fromkeys(fingers, 0.0),
            "fist": dict.fromkeys(fingers, 1.2),
            "pinch": {"left_thumb": 0.9, "left_index": 0.9, "left_middle": 0.0},
        },
        max_speed_rad_s=2.0,
        goal_tolerance_rad=0.02,
        settle_time_s=0.1,
    )
    return _Robot(
        frames=["world", "base_heading"],
        # No controller combination is validated: the base and the hand run one at a time.
        concurrent=[],
        bindings=[
            ProfileBinding(
                profile="base.planar_velocity/1",
                target="base",
                role="mobile_base",
                resources=["base"],
                modes=["goal", "stream"],
                controller="fake_base_velocity",
                assistance="none simulated: the commanded velocity is applied directly, with no gait or balance",
                stop_deadline_wall_s=1.0,
                limits=base,
            ),
            ProfileBinding(
                profile="hand.posture/1",
                target="left_hand",
                role="left_hand",
                resources=["left_hand"],
                modes=["goal"],
                controller="fake_left_hand_position",
                assistance="none: joints move to the named posture at a fixed speed",
                stop_deadline_wall_s=0.5,
                limits=hand,
            ),
        ],
        groups=[
            _Group(
                spec=ResourceSpec(id="base", role="mobile_base", joints=[], controller="fake_base_velocity"),
                positions={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                velocities=dict(ZERO_VELOCITY),
                hold=dict(ZERO_VELOCITY),
                still_speed={
                    "forward_m_s": base.stop_speed_m_s,
                    "left_m_s": base.stop_speed_m_s,
                    "yaw_rate_rad_s": base.stop_yaw_rate_rad_s,
                },
                settle_time_s=base.settle_time_s,
            ),
            _joint_group("left_hand", "left_hand", dict.fromkeys(fingers, 0.0), 0.05, hand.settle_time_s),
        ],
    )


def _arm() -> _Robot:
    # The joints, ranges, and home pose of the Franka Emika Panda that the mujoco backend's `panda` rig simulates
    # (panda.xml in MuJoCo Menagerie's franka_emika_panda), so a command for one works on the other.
    ranges = {
        "joint1": (-2.8973, 2.8973),
        "joint2": (-1.7628, 1.7628),
        "joint3": (-2.8973, 2.8973),
        "joint4": (-3.0718, -0.0698),
        "joint5": (-2.8973, 2.8973),
        "joint6": (-0.0175, 3.7525),
        "joint7": (-2.8973, 2.8973),
    }
    home = {
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -1.57079,
        "joint5": 0.0,
        "joint6": 1.57079,
        "joint7": -0.7853,
    }
    arm = JointTrajectoryLimits(
        profile="joints.trajectory/1",
        joints=[
            JointLimit(
                name=name,
                unit="rad",
                min_position=low,
                max_position=high,
                max_velocity=2.0,
                max_acceleration=8.0,
            )
            for name, (low, high) in ranges.items()
        ],
        max_points=64,
        max_duration_s=30.0,
        start_position_tolerance=0.05,
        start_velocity_tolerance=0.1,
        path_tolerance=0.1,
        goal_tolerance=0.02,
        settle_velocity=0.05,
        settle_time_s=0.2,
    )
    gripper = GripperWidthLimits(
        profile="gripper.width/1",
        min_width_m=0.0,
        max_width_m=0.08,
        speed_m_s=0.1,
        goal_tolerance_m=0.002,
        settle_time_s=0.1,
        # No contact or force detector exists here, so only width completion is offered.
        complete_on=["width_reached"],
    )
    return _Robot(
        frames=["world", "arm_base"],
        concurrent=[["arm", "gripper"]],
        bindings=[
            ProfileBinding(
                profile="joints.trajectory/1",
                target="arm",
                role="arm",
                resources=["arm"],
                modes=["goal"],
                controller="fake_arm_position",
                assistance="none: joints replay the validated trajectory",
                stop_deadline_wall_s=0.5,
                limits=arm,
            ),
            ProfileBinding(
                profile="gripper.width/1",
                target="gripper",
                role="gripper",
                resources=["gripper"],
                modes=["goal"],
                controller="fake_gripper_position",
                assistance="none: the jaw moves at a fixed speed",
                stop_deadline_wall_s=0.5,
                limits=gripper,
            ),
        ],
        groups=[
            _joint_group("arm", "arm", home, arm.settle_velocity, arm.settle_time_s),
            # One prismatic jaw whose position is the gap between the fingers, in metres.
            _joint_group("gripper", "gripper", {"jaw": gripper.max_width_m}, 0.005, gripper.settle_time_s),
        ],
    )


ROBOTS = {"g1": _g1, "arm": _arm}


def robots() -> dict[str, Backend]:
    """The robots `--backend fake` serves."""
    clock = SystemClock()
    return {"fake-g1": FakeBackend("g1", clock), "fake-arm": FakeBackend("arm", clock)}


class FakeBackend(Backend):
    def __init__(self, robot: str, clock: Clock):
        if robot not in ROBOTS:
            raise ValueError(f"unknown fake robot {robot!r}; choose from {sorted(ROBOTS)}")
        self._name = robot
        self._clock = clock
        self._last = clock.now()
        # A world lives in this process and names its clock's epoch, so the name must not repeat after a restart.
        self._instance = secrets.token_hex(4)
        self._world = 0
        self.faults: set[tuple[str, str]] = set()
        self.effects: list[tuple[Any, ...]] = []
        self._load()

    def _load(self) -> None:
        self._robot = ROBOTS[self._name]()
        self._groups = {group.spec.id: group for group in self._robot.groups}
        self._time = 0.0

    def _world_id(self) -> str:
        return f"fake-{self._name}-world-{self._instance}-{self._world}"

    def describe(self) -> BackendFacts:
        return BackendFacts(
            info=BackendInfo(kind="fake", physics="none", notice=NOTICE),
            world_id=self._world_id(),
            mode="fake",
            clock_mode="realtime",
            clocks=[
                ClockInfo(
                    id="fake_execution",
                    epoch=self._world_id(),
                    description="execution time of the fake, advanced from the host clock; zero at world reset",
                )
            ],
            frames=self._robot.frames,
            sensors=[
                SensorInfo(
                    id=rid,
                    kind="joint_state" if group.spec.joints else "base_state",
                    provenance="fake",
                    description=f"computed state of {rid}; not a measurement of anything real",
                )
                for rid, group in self._groups.items()
            ],
            revisions={"robot_description": f"fake-{self._name}/1", "calibration": "none"},
            resources=[group.spec for group in self._groups.values()],
            bindings=self._robot.bindings,
            concurrent=self._robot.concurrent,
            management=["recover", "world.reset"],
            period_s=PERIOD_S,
        )

    def apply(self, activity_id: str, contract: Contract) -> None:
        group = self._groups[contract.resources[0]]
        action, limits = contract.action, contract.limits
        self.effects.append(("apply", activity_id, action.profile, group.spec.id))
        judge: _Judge
        if isinstance(action, JointTrajectoryGoal):
            judge = TrajectoryJudge(action, limits)
        elif isinstance(action, GripperWidthGoal):
            judge = WidthJudge(action, limits)
        elif isinstance(action, HandPostureGoal):
            judge = PostureJudge(action, limits)
        else:
            judge = VelocityJudge(action, limits, self._time, _pose(group), action.duration_s)
        group.motion = _Motion(activity_id, self._time, judge)

    def track(self, activity_id: str, target: str, setpoint: BaseVelocityTarget, limits: BaseVelocityLimits) -> None:
        # Every target of the fake is its own resource.
        group = self._groups[target]
        self.effects.append(("track", activity_id, setpoint.forward_m_s, setpoint.left_m_s, setpoint.yaw_rate_rad_s))
        motion = group.motion
        if motion is not None and motion.activity_id == activity_id:
            motion.judge.command = setpoint  # replaced in place: no stop between updates
        else:
            group.motion = _Motion(activity_id, self._time, VelocityJudge(setpoint, limits, self._time, _pose(group)))

    def stop(self, resources: Sequence[str]) -> None:
        if any(("stop_lost", rid) in self.faults for rid in resources):
            raise ConnectionError("injected fault: the stop command was not delivered")
        self.effects.append(("stop", tuple(sorted(resources))))
        for rid in resources:
            self._halt(self._groups[rid])

    def _halt(self, group: _Group) -> None:
        group.motion = None
        group.velocities = dict.fromkeys(group.velocities, 0.0)
        if group.spec.joints:
            group.hold = dict(group.positions)
        # Standstill has to be measured again after the command.
        group.still_for = 0.0

    def manage(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
        self.effects.append((operation,))
        if operation == "recover":
            for rid in parameters["resources"]:
                self._halt(self._groups[rid])
            return {}
        if operation == "world.reset":
            self._world += 1
            self._load()
            return {"world_id": self._world_id()}
        raise RobotError("unsupported_operation", f"the fake backend has no {operation}", operation=operation)

    def advance(self) -> Feedback:
        now = self._clock.now()
        dt = max(0.0, now - self._last)
        self._last = now
        self._time += dt
        feedback = Feedback(execution_time_s=self._time, samples={}, lag_s=0.0)
        for rid, group in self._groups.items():
            motion = group.motion
            if motion is not None and motion.failure is None:
                _move(group, motion.judge, self._time - motion.started, dt, ("blocked", rid) in self.faults)
            still = all(abs(speed) <= group.still_speed[name] for name, speed in group.velocities.items())
            group.still_for = group.still_for + dt if still else 0.0
            if motion is not None:
                status = motion.failure or self._judge(group, motion, dt)
                if status.phase == "failed":
                    motion.failure = status
                    group.velocities = dict.fromkeys(group.velocities, 0.0)
                elif status.phase == "done":
                    group.motion = None
                    group.hold.update(_reached(motion.judge))
                feedback.activities[motion.activity_id] = status

            lost = ("feedback_loss", rid) in self.faults
            if not lost or group.sample is None:
                values: dict[str, Any] = {"positions": dict(group.positions), "velocities": dict(group.velocities)}
                if "jaw" in group.positions:
                    values["width_m"] = group.positions["jaw"]
                if "x_m" in group.positions:
                    values["frame_id"] = "world"  # the frame of the base pose
                group.sample = Sample(
                    source=f"fake:{rid}",
                    provenance="fake",
                    clock_id="fake_execution",
                    clock_epoch=self._world_id(),
                    capture_time_ns=str(round(self._time * 1e9)),
                    valid=True,
                    values=values,
                )
            # A lost sensor repeats its last reading, marked invalid; it never confirms standstill.
            feedback.samples[rid] = group.sample.model_copy(update={"valid": False}) if lost else group.sample
            feedback.stopped[rid] = not lost and group.motion is None and group.still_for >= group.settle_time_s
            feedback.holding[rid] = dict(group.hold)
        return feedback

    def _judge(self, group: _Group, motion: _Motion, dt: float) -> ActivityStatus:
        """Hand the profile's judge what this group measures now."""
        judge = motion.judge
        if isinstance(judge, TrajectoryJudge):
            joints = judge.action.joints
            positions, velocities = [group.positions[n] for n in joints], [group.velocities[n] for n in joints]
            return judge.update(self._time - motion.started, positions, velocities, dt)
        if isinstance(judge, WidthJudge):
            return judge.update(group.positions["jaw"], dt)
        if isinstance(judge, PostureJudge):
            return judge.update(group.positions, dt)
        return judge.update(self._time, _pose(group), group.still_for)


def _pose(group: _Group) -> tuple[float, float, float]:
    return group.positions["x_m"], group.positions["y_m"], group.positions["yaw_rad"]


def _reached(judge: _Judge) -> dict[str, float]:
    """What a group holds after its goal is met: the goal's own end point. A base holds zero velocity."""
    if isinstance(judge, TrajectoryJudge):
        return dict(zip(judge.action.joints, judge.action.points[-1].positions, strict=True))
    if isinstance(judge, WidthJudge):
        return {"jaw": judge.action.width_m}
    if isinstance(judge, PostureJudge):
        return dict(judge.limits.postures[judge.action.posture])
    return {}


def _move(group: _Group, judge: _Judge, elapsed: float, dt: float, blocked: bool) -> None:
    """The fake's controllers: joints follow their command exactly, unless blocked. Nothing is simulated."""
    if isinstance(judge, TrajectoryJudge):
        reference, rates = sample_trajectory(judge.action.points, elapsed)
        for name, position, rate in zip(judge.action.joints, reference, rates, strict=True):
            if not blocked:
                group.positions[name] = position
            group.velocities[name] = 0.0 if blocked else rate
    elif isinstance(judge, VelocityJudge):  # a timed goal, or a stream's setpoint
        command, duration = judge.command, judge.duration_s
        timed = duration is not None
        # Only the part of this step that lies inside the goal's duration is travelled.
        travel = min(dt, max(0.0, duration - (elapsed - dt))) if timed else dt
        travelling = not timed or elapsed < duration
        moved = (0.0, 0.0, 0.0) if blocked else (command.forward_m_s, command.left_m_s, command.yaw_rate_rad_s)
        forward, left, yaw_rate = moved
        yaw = group.positions["yaw_rad"]
        group.positions["x_m"] += (forward * math.cos(yaw) - left * math.sin(yaw)) * travel
        group.positions["y_m"] += (forward * math.sin(yaw) + left * math.cos(yaw)) * travel
        group.positions["yaw_rad"] += yaw_rate * travel
        group.velocities = dict(zip(ZERO_VELOCITY, moved if travelling else (0.0, 0.0, 0.0), strict=True))
    else:  # gripper.width/1 or hand.posture/1: each joint moves toward its target at a fixed speed
        if isinstance(judge, WidthJudge):
            targets, speed = {"jaw": judge.action.width_m}, judge.limits.speed_m_s
        else:
            targets, speed = judge.limits.postures[judge.action.posture], judge.limits.max_speed_rad_s
        for name, goal in targets.items():
            step = 0.0 if blocked else max(-speed * dt, min(speed * dt, goal - group.positions[name]))
            group.positions[name] += step
            group.velocities[name] = step / dt if dt > 0 else 0.0
