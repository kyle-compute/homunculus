"""What each motion profile means, independent of any backend.

A request is resolved against the binding a backend declared: unsupported
parameters and out-of-domain values are rejected, and omitted settings become
concrete values in the accepted contract. This module also owns the
``joints.trajectory/1`` interpolation rule, so admission checks and every
backend controller evaluate the same path.

Whether a goal or stream succeeded is decided here too, by one judge per
profile. A backend commands its robot and feeds the judge what it measured;
the judge applies the binding's tolerances and settle times and returns the
ActivityStatus the backend reports. Every backend therefore succeeds and fails
by the same rules, with the same measurements.

Each profile is one `Profile` in `PROFILES`: its request and limits models, its
admission rule, and its judge. A new kind of motion is a new `Profile` passed to
`register()`; the runtime, the service and the client look profiles up there.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Any, Concatenate, Literal, get_args

from .models import (
    BaseVelocityGoal,
    BaseVelocityLimits,
    BaseVelocityTarget,
    GripperWidthGoal,
    GripperWidthLimits,
    HandPostureGoal,
    HandPostureLimits,
    JointLimit,
    JointTrajectoryGoal,
    JointTrajectoryLimits,
    Model,
    ProfileBinding,
    RobotError,
    StreamSpec,
    ToolPoseGoal,
    ToolPoseLimits,
    TrajectoryPoint,
)


def find_binding(bindings: Iterable[ProfileBinding], profile: str, target: str, mode: str) -> ProfileBinding:
    """Return the binding for a profile and target, or say exactly what is missing."""
    bindings = list(bindings)
    offered = [b for b in bindings if b.profile == profile]
    if not offered:
        raise RobotError(
            "unsupported_profile",
            f"this robot does not support {profile}",
            profile=profile,
            supported=sorted({b.profile for b in bindings}),
        )
    for binding in offered:
        if binding.target == target:
            if mode not in binding.modes:
                raise RobotError(
                    "unsupported_mode",
                    f"{profile} on {target!r} supports {' and '.join(binding.modes)} execution, not {mode}",
                    profile=profile,
                    target=target,
                    modes=binding.modes,
                )
            return binding
    raise RobotError(
        "unknown_target",
        f"{profile} is not bound to a target named {target!r}",
        profile=profile,
        targets=[b.target for b in offered],
    )


def _resolve_velocity_goal(action: BaseVelocityGoal, limits: BaseVelocityLimits) -> BaseVelocityGoal:
    check_velocity(action, limits)
    _within("duration_s", action.duration_s, 0.0, limits.max_goal_duration_s)
    _check_verifiable(action, limits)
    return action


def _resolve_posture(action: HandPostureGoal, limits: HandPostureLimits) -> HandPostureGoal:
    if action.posture not in limits.postures:
        raise RobotError(
            "unsupported_parameter",
            f"posture {action.posture!r} is not advertised for {action.target!r}",
            parameter="posture",
            supported=sorted(limits.postures),
        )
    return action


def _resolve_width(action: GripperWidthGoal, limits: GripperWidthLimits) -> GripperWidthGoal:
    if action.complete_on not in limits.complete_on:
        raise RobotError(
            "unsupported_parameter",
            f"{action.target!r} has no qualified detector for complete_on={action.complete_on}",
            parameter="complete_on",
            supported=limits.complete_on,
        )
    for name, rule, advertised in (
        ("contact_rule", action.contact_rule, limits.contact_rules),
        ("force_rule", action.force_rule, limits.force_rules),
    ):
        if rule is not None and rule not in advertised:
            raise RobotError(
                "unsupported_parameter",
                f"{name} {rule!r} is not an advertised rule for {action.target!r}",
                parameter=name,
                supported=advertised,
            )
    _within("width_m", action.width_m, limits.min_width_m, limits.max_width_m)
    return action


QUATERNION_NORM_TOLERANCE = 1e-3


def _resolve_pose(action: ToolPoseGoal, limits: ToolPoseLimits) -> ToolPoseGoal:
    if action.frame_id not in limits.frames:
        raise RobotError(
            "unsupported_parameter",
            f"frame {action.frame_id!r} is not offered for {action.target!r}",
            parameter="frame_id",
            supported=limits.frames,
        )
    norm = math.sqrt(sum(v * v for v in action.orientation_xyzw))
    if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise RobotError(
            "invalid_request",
            f"orientation_xyzw has length {norm:.4f}; "
            f"a unit quaternion is required (within {QUATERNION_NORM_TOLERANCE})",
            parameter="orientation_xyzw",
        )
    for axis, value, low, high in zip(
        "xyz", action.position_m, limits.workspace_min_m, limits.workspace_max_m, strict=True
    ):
        _within(f"position_m.{axis}", value, low, high)
    speed = limits.default_speed_m_s if action.speed_m_s is None else action.speed_m_s
    _within("speed_m_s", speed, limits.min_speed_m_s, limits.max_speed_m_s)
    position_tolerance = action.position_tolerance_m or limits.position_tolerance_m
    orientation_tolerance = action.orientation_tolerance_rad or limits.orientation_tolerance_rad
    _within("position_tolerance_m", position_tolerance, 0.0, limits.max_position_tolerance_m)
    _within("orientation_tolerance_rad", orientation_tolerance, 0.0, limits.max_orientation_tolerance_rad)
    return action.model_copy(
        update={
            "orientation_xyzw": [v / norm for v in action.orientation_xyzw],
            "speed_m_s": speed,
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_rad": orientation_tolerance,
        }
    )


def resolve_stream(spec: StreamSpec, binding: ProfileBinding) -> BaseVelocityLimits:
    """Check a stream's fixed settings and first target; return the limits it runs under."""
    limits = binding.limits
    if not isinstance(limits, BaseVelocityLimits) or limits.min_update_validity_wall_s is None:
        raise RobotError("unsupported_mode", f"{spec.profile} has no stream form here", profile=spec.profile)
    _within(
        "update_validity_wall_s",
        spec.update_validity_wall_s,
        limits.min_update_validity_wall_s,
        limits.max_update_validity_wall_s,
    )
    check_velocity(spec.initial_target, limits)
    return limits


def check_velocity(target: BaseVelocityTarget, limits: BaseVelocityLimits) -> None:
    if target.frame_id not in limits.frames:
        raise RobotError(
            "unsupported_parameter",
            f"velocity frame {target.frame_id!r} is not available on this base",
            parameter="frame_id",
            supported=limits.frames,
        )
    _within("forward_m_s", target.forward_m_s, -limits.max_forward_m_s, limits.max_forward_m_s)
    _within("left_m_s", target.left_m_s, -limits.max_left_m_s, limits.max_left_m_s)
    _within("yaw_rate_rad_s", target.yaw_rate_rad_s, -limits.max_yaw_rate_rad_s, limits.max_yaw_rate_rad_s)


def off_track(measured: Sequence[float], command: Sequence[float], limits: BaseVelocityLimits) -> bool:
    """True when a measured (forward, left, yaw rate) is outside the tolerance around this command on any axis."""
    ratio = limits.max_tracking_error_ratio
    absolute = (limits.max_tracking_error_m_s, limits.max_tracking_error_m_s, limits.max_tracking_error_rad_s)
    return any(
        abs(value - wanted) > max(floor, ratio * abs(wanted))
        for value, wanted, floor in zip(measured, command, absolute, strict=True)
    )


def _check_verifiable(goal: BaseVelocityGoal, limits: BaseVelocityLimits) -> None:
    """Refuse timed travel whose success could not be told apart from a base that never moved."""
    command = (goal.forward_m_s, goal.left_m_s, goal.yaw_rate_rad_s)
    if not any(command):
        return  # standing still for a while is a goal that a still base does meet
    if not off_track((0.0, 0.0, 0.0), command, limits):
        raise RobotError(
            "limit_exceeded",
            "this velocity is within the tracking tolerance of standing still, so its travel could not be "
            f"verified; command more than {limits.max_tracking_error_m_s:g} m/s "
            f"or {limits.max_tracking_error_rad_s:g} rad/s",
            parameter="velocity",
            minimum_m_s=limits.max_tracking_error_m_s,
            minimum_rad_s=limits.max_tracking_error_rad_s,
        )
    if goal.duration_s <= limits.transient_s:
        raise RobotError(
            "limit_exceeded",
            f"duration_s={goal.duration_s:g} ends within the {limits.transient_s:g} s startup transient, "
            "before tracking can be measured; ask for longer travel",
            parameter="duration_s",
            value=goal.duration_s,
            minimum=limits.transient_s,
        )


def _within(parameter: str, value: float, low: float, high: float, **where: object) -> None:
    if not low <= value <= high:
        raise RobotError(
            "limit_exceeded",
            f"{parameter}={value:g} is outside the supported range {low:g} to {high:g}",
            parameter=parameter,
            value=value,
            minimum=low,
            maximum=high,
            **where,
        )


# --------------------------------------------------------------------------
# joints.trajectory/1
# --------------------------------------------------------------------------


def _resolve_trajectory(action: JointTrajectoryGoal, limits: JointTrajectoryLimits) -> JointTrajectoryGoal:
    known = {joint.name: joint for joint in limits.joints}
    unknown = [name for name in action.joints if name not in known]
    if unknown:
        raise RobotError(
            "unsupported_parameter",
            f"{action.target!r} has no joint named {', '.join(map(repr, unknown))}; it has {', '.join(known)}",
            parameter="joints",
            supported=sorted(known),
        )
    if len(set(action.joints)) != len(action.joints):
        raise RobotError("invalid_request", "each joint may be named only once", parameter="joints")
    points = action.points
    if len(points) > limits.max_points:
        raise RobotError(
            "limit_exceeded",
            f"{len(points)} waypoints exceed the limit of {limits.max_points}",
            parameter="points",
            maximum=limits.max_points,
        )
    width = len(action.joints)
    for index, point in enumerate(points):
        if len(point.positions) != width or len(point.velocities) != width:
            raise RobotError(
                "invalid_request",
                f"waypoint {index} must give one position and one velocity per joint",
                point=index,
            )
        if index and point.time_s <= points[index - 1].time_s:
            raise RobotError("invalid_request", "waypoint times must strictly increase", point=index)
    _within("points[-1].time_s", points[-1].time_s, 0.0, limits.max_duration_s)
    if any(abs(v) > limits.settle_velocity for v in points[-1].velocities):
        raise RobotError(
            "limit_exceeded",
            f"after=hold needs the final waypoint nearly at rest: each velocity within {limits.settle_velocity:g}",
            parameter="points[-1].velocities",
            maximum=limits.settle_velocity,
        )

    # A client may tighten a tolerance; it may not relax the binding's.
    tolerances = {}
    for name in ("path_tolerance", "goal_tolerance"):
        requested, configured = getattr(action, name), getattr(limits, name)
        if requested is not None and requested > configured:
            raise RobotError(
                "limit_exceeded",
                f"{name}={requested:g} would relax the configured {configured:g}",
                parameter=name,
                maximum=configured,
            )
        tolerances[name] = configured if requested is None else requested

    joints = [known[name] for name in action.joints]
    check_path(points, joints)
    return action.model_copy(update=tolerances)


def verified_joint_start(
    joints: Sequence[str],
    positions: Mapping[str, float],
    velocities: Mapping[str, float],
    *,
    path_tolerance: float,
    settle_velocity: float,
    reference: JointReference | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Anchor a joint motion to the controller reference only after verifying it against measurements.

    Refuses partial-joint handoffs that leave any moving joint uncommanded, and refuses references
    that have diverged beyond `path_tolerance` from the real sensor reading.
    """
    if reference is not None:
        commanded = set(joints)
        for name, meas_v in velocities.items():
            if name in commanded:
                continue
            ref_v = reference.velocities.get(name, 0.0)
            if abs(meas_v) > settle_velocity or abs(ref_v) > settle_velocity:
                raise RobotError(
                    "start_condition_failed",
                    f"joint {name!r} is still moving ({max(abs(meas_v), abs(ref_v)):.3f}) "
                    "and is not commanded by this goal",
                    joint=name,
                    measured_velocity=meas_v,
                    reference_velocity=ref_v,
                    settle_velocity=settle_velocity,
                )
    start_q: list[float] = []
    start_v: list[float] = []
    start_a: list[float] = []
    for name in joints:
        meas_q, meas_v = positions[name], velocities[name]
        if reference is not None and name in reference.positions:
            ref_q = reference.positions[name]
            err = abs(ref_q - meas_q)
            if err > path_tolerance:
                raise RobotError(
                    "start_condition_failed",
                    f"controller reference for {name!r} diverges from measured position by {err:.4f} "
                    f"(limit {path_tolerance:g})",
                    joint=name,
                    reference_position=ref_q,
                    measured_position=meas_q,
                    path_tolerance=path_tolerance,
                )
            start_q.append(ref_q)
            start_v.append(reference.velocities.get(name, meas_v))
            start_a.append(reference.accelerations.get(name, 0.0))
        else:
            start_q.append(meas_q)
            start_v.append(meas_v)
            start_a.append(0.0)
    return start_q, start_v, start_a


def activate_trajectory(
    action: JointTrajectoryGoal,
    limits: JointTrajectoryLimits,
    positions: Mapping[str, float],
    velocities: Mapping[str, float],
    *,
    reference: JointReference | None = None,
) -> JointTrajectoryGoal:
    """Bind a resolved trajectory to the measured state (or verified controller reference) at activation.

    A waypoint at time zero must match the start state. Otherwise the start state becomes the waypoint
    at zero and that first segment is validated like any other, before any movement.
    """
    start_q, start_v, _ = verified_joint_start(
        action.joints,
        positions,
        velocities,
        path_tolerance=action.path_tolerance or limits.path_tolerance,
        settle_velocity=limits.settle_velocity,
        reference=reference,
    )
    first = action.points[0]
    if first.time_s == 0.0:
        for name, q, v, want_q, want_v in zip(
            action.joints, start_q, start_v, first.positions, first.velocities, strict=True
        ):
            if abs(q - want_q) > limits.start_position_tolerance or abs(v - want_v) > limits.start_velocity_tolerance:
                raise RobotError(
                    "start_condition_failed",
                    f"{name} is at {q:.4f} moving {v:.4f}; "
                    f"the waypoint at time zero expects {want_q:.4f} and {want_v:.4f}",
                    joint=name,
                    measured_position=q,
                    measured_velocity=v,
                    position_tolerance=limits.start_position_tolerance,
                    velocity_tolerance=limits.start_velocity_tolerance,
                )
        return action
    start = TrajectoryPoint(time_s=0.0, positions=start_q, velocities=start_v)
    known = {joint.name: joint for joint in limits.joints}
    check_path(
        [start, first],
        [known[name] for name in action.joints],
        slack=limits.start_position_tolerance,
    )
    return action.model_copy(update={"points": [start, *action.points]})


def check_path(points: Sequence[TrajectoryPoint], joints: Sequence[JointLimit], slack: float = 0.0) -> None:
    """Reject a path whose interpolated position, velocity, or acceleration leaves the limits."""
    for index, point in enumerate(points):
        for joint, q, v in zip(joints, point.positions, point.velocities, strict=True):
            _within(f"{joint.name} position", q, joint.min_position - slack, joint.max_position + slack, point=index)
            _within(f"{joint.name} velocity", v, -joint.max_velocity, joint.max_velocity, point=index)
    for index, (a, b) in enumerate(itertools.pairwise(points)):
        duration = b.time_s - a.time_s
        for joint, p0, v0, p1, v1 in zip(joints, a.positions, a.velocities, b.positions, b.velocities, strict=True):
            low, high, speed, acceleration = _segment_extrema(p0, v0, p1, v1, duration)
            where = {"segment": index}
            _within(f"{joint.name} position", low, joint.min_position - slack, math.inf, **where)
            _within(f"{joint.name} position", high, -math.inf, joint.max_position + slack, **where)
            _within(f"{joint.name} speed", speed, 0.0, joint.max_velocity, **where)
            _within(f"{joint.name} acceleration", acceleration, 0.0, joint.max_acceleration, **where)


def sample_trajectory(points: Sequence[TrajectoryPoint], time_s: float) -> tuple[list[float], list[float]]:
    """Reference positions and velocities at an execution time since activation."""
    if time_s <= points[0].time_s:
        return list(points[0].positions), list(points[0].velocities)
    if time_s >= points[-1].time_s:
        return list(points[-1].positions), [0.0] * len(points[-1].positions)
    later = bisect.bisect_right(points, time_s, key=lambda point: point.time_s)  # the first waypoint after time_s
    a, b = points[later - 1], points[later]
    duration = b.time_s - a.time_s
    s = (time_s - a.time_s) / duration
    positions, velocities = [], []
    for p0, v0, p1, v1 in zip(a.positions, a.velocities, b.positions, b.velocities, strict=True):
        c3, c2, c1, c0 = _cubic(p0, v0, p1, v1, duration)
        positions.append(((c3 * s + c2) * s + c1) * s + c0)
        velocities.append((3 * c3 * s * s + 2 * c2 * s + c1) / duration)
    return positions, velocities


def _cubic(p0: float, v0: float, p1: float, v1: float, duration: float) -> tuple[float, float, float, float]:
    """Coefficients of the Hermite segment p(s) = c3 s^3 + c2 s^2 + c1 s + c0, s in [0, 1]."""
    c1 = v0 * duration
    c3 = 2 * p0 - 2 * p1 + c1 + v1 * duration
    c2 = -3 * p0 + 3 * p1 - 2 * c1 - v1 * duration
    return c3, c2, c1, p0


def _segment_extrema(p0: float, v0: float, p1: float, v1: float, duration: float) -> tuple[float, float, float, float]:
    """Exact (min position, max position, peak speed, peak acceleration) of one segment."""
    c3, c2, c1, c0 = _cubic(p0, v0, p1, v1, duration)

    def position(s: float) -> float:
        return ((c3 * s + c2) * s + c1) * s + c0

    def speed(s: float) -> float:
        return abs(3 * c3 * s * s + 2 * c2 * s + c1) / duration

    # Position turns where velocity is zero; speed peaks where acceleration is zero.
    turning = [0.0, 1.0]
    a, b, c = 3 * c3, 2 * c2, c1
    if abs(a) > 1e-12:
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            root = math.sqrt(discriminant)
            turning += [s for s in ((-b - root) / (2 * a), (-b + root) / (2 * a)) if 0 < s < 1]
    elif abs(b) > 1e-12 and 0 < -c / b < 1:
        turning.append(-c / b)
    values = [position(s) for s in turning]

    speeds = [speed(0.0), speed(1.0)]
    if abs(c3) > 1e-12 and 0 < -c2 / (3 * c3) < 1:
        speeds.append(speed(-c2 / (3 * c3)))
    # Divided twice: the square of a very short duration is zero, where the quotient is merely too large.
    acceleration = max(abs(2 * c2), abs(6 * c3 + 2 * c2)) / duration / duration
    return min(values), max(values), max(speeds), acceleration


# --------------------------------------------------------------------------
# Judges: success and failure, decided from what a backend measured
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityStatus:
    """Progress of one goal or stream, as a judge decided it from measurements."""

    phase: Literal["executing", "finishing", "done", "failed"]
    measurements: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    detail: str | None = None


class MotionJudge:
    """Owns the latest verdict computed from measurements, including readings fed by a backend worker."""

    def __init__(self) -> None:
        self._status: ActivityStatus | None = None
        self._lock = Lock()

    @property
    def status(self) -> ActivityStatus | None:
        with self._lock:
            return self._status


def measures[J: MotionJudge, **P](
    update: Callable[Concatenate[J, P], ActivityStatus],
) -> Callable[Concatenate[J, P], ActivityStatus]:
    """Decorate a judge's update so its verdict is published as the judge's `status` once computed: a backend
    that feeds the judge itself, and the runtime, read it there."""

    @wraps(update)
    def measure(judge: J, /, *args: P.args, **kwargs: P.kwargs) -> ActivityStatus:
        outcome = update(judge, *args, **kwargs)
        with judge._lock:
            judge._status = outcome
        return outcome

    return measure


class TrajectoryJudge(MotionJudge):
    """joints.trajectory/1. The measured path must stay within path_tolerance of the reference. After the last
    waypoint, the goal is met once every joint has been within goal_tolerance and slower than settle_velocity
    for settle_time_s."""

    def __init__(self, action: JointTrajectoryGoal, limits: JointTrajectoryLimits):
        super().__init__()
        self.action, self.limits = action, limits
        self.settled_for = 0.0
        self.peak_error = 0.0

    @measures
    def update(
        self, elapsed: float, positions: Sequence[float], velocities: Sequence[float], dt: float
    ) -> ActivityStatus:
        """Judge positions and velocities measured `elapsed` seconds after activation, in the order of the
        action's joints. `dt` is the time since the previous update."""
        action, limits = self.action, self.limits
        reference, _ = sample_trajectory(action.points, elapsed)
        errors = [abs(q - r) for q, r in zip(positions, reference, strict=True)]
        worst = max(range(len(errors)), key=errors.__getitem__)
        self.peak_error = max(self.peak_error, errors[worst])
        if errors[worst] > action.path_tolerance:
            measurements = {
                "joint": action.joints[worst],
                "tracking_error": errors[worst],
                "path_tolerance": action.path_tolerance,
                "at_s": elapsed,
            }
            detail = "measured position left the tolerance around the reference path"
            return ActivityStatus("failed", measurements, error="path_tolerance_violated", detail=detail)
        if elapsed < action.points[-1].time_s:
            return ActivityStatus("executing")
        slow = all(abs(v) <= limits.settle_velocity for v in velocities)
        self.settled_for = self.settled_for + dt if slow and errors[worst] <= action.goal_tolerance else 0.0
        if self.settled_for < limits.settle_time_s:
            return ActivityStatus("finishing")
        measurements = {
            "final_error": dict(zip(action.joints, errors, strict=True)),
            "peak_tracking_error": self.peak_error,
            "goal_tolerance": action.goal_tolerance,
        }
        return ActivityStatus("done", measurements)


class WidthJudge(MotionJudge):
    """gripper.width/1 with complete_on=width_reached: met once the measured gap has stayed within goal_tolerance_m
    of the request for settle_time_s. Nothing here tells a blocked jaw from a slow one, so a jaw that never gets
    there does not fail here; the goal's deadline ends it."""

    def __init__(self, action: GripperWidthGoal, limits: GripperWidthLimits):
        super().__init__()
        self.action, self.limits = action, limits
        self.within_for = 0.0

    @measures
    def update(self, width_m: float, dt: float) -> ActivityStatus:
        error = abs(width_m - self.action.width_m)
        if error > self.limits.goal_tolerance_m:
            self.within_for = 0.0
            return ActivityStatus("executing")
        self.within_for += dt
        if self.within_for < self.limits.settle_time_s:
            return ActivityStatus("finishing")
        return ActivityStatus("done", {"width_m": width_m, "error_m": error})


class PostureJudge(MotionJudge):
    """hand.posture/1: met once every joint the posture names has stayed within goal_tolerance_rad of it for
    settle_time_s. As with a gripper, a hand that never gets there is ended by the goal's deadline."""

    def __init__(self, action: HandPostureGoal, limits: HandPostureLimits):
        super().__init__()
        self.action, self.limits = action, limits
        self.within_for = 0.0

    @measures
    def update(self, positions: Mapping[str, float], dt: float) -> ActivityStatus:
        targets = self.limits.postures[self.action.posture]
        error = max(abs(positions[name] - angle) for name, angle in targets.items())
        if error > self.limits.goal_tolerance_rad:
            self.within_for = 0.0
            return ActivityStatus("executing")
        self.within_for += dt
        if self.within_for < self.limits.settle_time_s:
            return ActivityStatus("finishing")
        return ActivityStatus("done", {"posture": self.action.posture, "max_error_rad": error})


class PoseJudge(MotionJudge):
    """tool.pose.linear/1. The reference moves the tool origin along the straight segment from where it was
    measured at activation to the goal, and turns it along the shortest rotation, both on a smoothstep
    (zero speed at both ends). It lasts long enough that neither the peak speed nor the peak rotation rate
    exceeds what the goal and binding allow.

    While the reference moves, the measured origin must stay within path_tolerance_m of the segment. After
    it ends, the goal is met once the measured pose has been within both goal tolerances for settle_time_s,
    and fails if that has not happened within max_settle_wait_s.
    """

    def __init__(
        self, action: ToolPoseGoal, limits: ToolPoseLimits, start_m: Sequence[float], start_xyzw: Sequence[float]
    ):
        super().__init__()
        self.action, self.limits = action, limits
        self.start_m, self.start_xyzw = list(start_m), _same_hemisphere(start_xyzw, action.orientation_xyzw)
        distance = math.dist(self.start_m, action.position_m)
        turn = quaternion_angle(self.start_xyzw, action.orientation_xyzw)
        # A smoothstep's peak rate is 1.5 times its mean.
        self.duration_s = max(1.5 * distance / action.speed_m_s, 1.5 * turn / limits.max_rotation_rate_rad_s, 0.2)
        self.within_for = 0.0
        self.peak_path_error = 0.0

    def reference(self, elapsed: float) -> tuple[list[float], list[float]]:
        """The reference position and xyzw orientation `elapsed` seconds after activation."""
        t = min(1.0, max(0.0, elapsed / self.duration_s))
        s = t * t * (3 - 2 * t)
        position = [a + s * (b - a) for a, b in zip(self.start_m, self.action.position_m, strict=True)]
        return position, _slerp(self.start_xyzw, self.action.orientation_xyzw, s)

    @measures
    def update(self, elapsed: float, position_m: Sequence[float], xyzw: Sequence[float], dt: float) -> ActivityStatus:
        action, limits = self.action, self.limits
        position_error = math.dist(position_m, action.position_m)
        orientation_error = quaternion_angle(xyzw, action.orientation_xyzw)
        path_error = _distance_to_segment(position_m, self.start_m, action.position_m)
        self.peak_path_error = (
            max(self.peak_path_error, path_error) if elapsed <= self.duration_s else self.peak_path_error
        )
        measurements = {
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "peak_path_error_m": self.peak_path_error,
            "position_m": list(position_m),
            "orientation_xyzw": list(xyzw),
        }
        if elapsed <= self.duration_s:
            if path_error > limits.path_tolerance_m:
                detail = "the tool left the tolerance around the straight segment; something may be in the way"
                measurements.update(path_error_m=path_error, path_tolerance_m=limits.path_tolerance_m, at_s=elapsed)
                return ActivityStatus("failed", measurements, error="path_tolerance_violated", detail=detail)
            return ActivityStatus("executing")
        within = position_error <= action.position_tolerance_m and orientation_error <= action.orientation_tolerance_rad
        # Only the part of this step after the reference ended counts toward settling.
        self.within_for = self.within_for + min(dt, elapsed - self.duration_s) if within else 0.0
        if self.within_for >= limits.settle_time_s:
            return ActivityStatus("done", measurements)
        if elapsed - self.duration_s >= limits.max_settle_wait_s:
            detail = (
                f"the tool did not hold within {action.position_tolerance_m:g} m and "
                f"{action.orientation_tolerance_rad:g} rad of the pose; "
                "it may be blocked, holding a load, or out of reach"
            )
            return ActivityStatus("failed", measurements, error="goal_not_reached", detail=detail)
        return ActivityStatus("finishing")


def quaternion_angle(a: Sequence[float], b: Sequence[float]) -> float:
    """The angle in radians of the rotation between two unit xyzw quaternions."""
    dot = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    return 2 * math.acos(min(1.0, dot))


def _same_hemisphere(q: Sequence[float], toward: Sequence[float]) -> list[float]:
    """q, or -q (the same rotation) if that is nearer `toward`, so interpolation takes the shorter way."""
    return [-v for v in q] if sum(x * y for x, y in zip(q, toward, strict=True)) < 0 else list(q)


def _slerp(a: Sequence[float], b: Sequence[float], s: float) -> list[float]:
    dot = min(1.0, sum(x * y for x, y in zip(a, b, strict=True)))
    if dot > 0.9995:  # nearly the same: linear interpolation is exact to this precision
        mixed = [x + s * (y - x) for x, y in zip(a, b, strict=True)]
    else:
        theta = math.acos(dot)
        wa, wb = math.sin((1 - s) * theta) / math.sin(theta), math.sin(s * theta) / math.sin(theta)
        mixed = [wa * x + wb * y for x, y in zip(a, b, strict=True)]
    norm = math.sqrt(sum(v * v for v in mixed))
    return [v / norm for v in mixed]


def _distance_to_segment(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    ab = [y - x for x, y in zip(a, b, strict=True)]
    length2 = sum(v * v for v in ab)
    t = (
        0.0
        if length2 == 0
        else min(1.0, max(0.0, sum((pi - ai) * v for pi, ai, v in zip(p, a, ab, strict=True)) / length2))
    )
    return math.dist(p, [ai + t * v for ai, v in zip(a, ab, strict=True)])


Pose = tuple[float, float, float]  # x_m, y_m, yaw_rad in the world frame


class VelocityJudge(MotionJudge):
    """base.planar_velocity/1, as a timed goal or as a stream.

    Velocity is measured from successive base poses, averaged over tracking_window_s. From transient_s after
    activation, it must stay within the tracking tolerance of the command (see off_track); an error lasting
    longer than max_tracking_error_duration_s fails. Elapsed time is not evidence: a timed goal must have been
    measured on track at least once, and is met only once the backend has measured the base at rest for
    settle_time_s after the travel.
    """

    def __init__(
        self,
        command: BaseVelocityTarget,
        limits: BaseVelocityLimits,
        time_s: float,
        pose: Pose,
        duration_s: float | None = None,
    ):
        super().__init__()
        self.command = command  # a stream replaces this with each new target
        self.limits = limits
        self.duration_s = duration_s  # None for a stream, which travels until it ends
        self.start = time_s
        self.origin = pose
        # (time, x, y, yaw) at each update, back to the newest one at least tracking_window_s old.
        self.poses = [(time_s, *pose)]
        self.turned = 0.0  # summed update by update, so it survives the wrap at pi
        self.travelled_s = 0.0
        self.measured = (0.0, 0.0, 0.0)
        self.peak = [0.0, 0.0, 0.0]
        self.off_track_for = 0.0
        self.tracked = not any(self._commanded())  # standing still is met by a base that stays still

    @measures
    def update(self, time_s: float, pose: Pose, still_for_s: float) -> ActivityStatus:
        """Judge the pose measured at `time_s`. `still_for_s` is how long the backend has measured the base at
        rest, up to now: the same standstill it reports when confirming a stop."""
        limits, command = self.limits, self._commanded()
        began = self.poses[-1][0] - self.start  # when the interval up to this pose began, since activation
        self.turned += _turn(self.poses[-1][3], pose[2])
        self.poses.append((time_s, *pose))
        if self.duration_s is None or began < self.duration_s - 1e-9:
            self.travelled_s = time_s - self.start
            self.measured = _mean_velocity(self.poses, limits.tracking_window_s)
            while len(self.poses) > 2 and time_s - self.poses[1][0] >= limits.tracking_window_s - 1e-9:
                del self.poses[0]
            if self.travelled_s >= limits.transient_s - 1e-9:
                self.peak = [max(p, abs(m - c)) for p, m, c in zip(self.peak, self.measured, command, strict=True)]
                if off_track(self.measured, command, limits):
                    self.off_track_for += time_s - self.poses[-2][0]
                else:
                    self.off_track_for, self.tracked = 0.0, True
            if self.off_track_for > limits.max_tracking_error_duration_s:
                return self._tracking_error()
            if self.duration_s is None or self.travelled_s < self.duration_s - 1e-9:
                return ActivityStatus("executing")
        if not self.tracked:
            return self._tracking_error()
        if still_for_s < limits.settle_time_s:
            return ActivityStatus("finishing")
        return ActivityStatus("done", {**self.measurements(), "peak_tracking_error": self._axes(self.peak)})

    def measurements(self) -> dict[str, Any]:
        """Where the base went: net displacement since activation, and how long it travelled."""
        x, y, _ = self.origin
        _, x_now, y_now, _ = self.poses[-1]
        travelled = self.travelled_s if self.duration_s is None else min(self.travelled_s, self.duration_s)
        return {
            "displacement": {"frame_id": "world", "x_m": x_now - x, "y_m": y_now - y, "yaw_rad": self.turned},
            "travel_time_s": travelled,
        }

    def _commanded(self) -> tuple[float, float, float]:
        return (self.command.forward_m_s, self.command.left_m_s, self.command.yaw_rate_rad_s)

    def _tracking_error(self) -> ActivityStatus:
        measurements = {
            "commanded": self._axes(self._commanded()),
            "measured": self._axes(self.measured),
            **self.measurements(),
        }
        detail = "the base did not follow the commanded velocity; it may be blocked"
        return ActivityStatus("failed", measurements, error="tracking_error", detail=detail)

    @staticmethod
    def _axes(values: Sequence[float]) -> dict[str, float]:
        return dict(zip(("forward_m_s", "left_m_s", "yaw_rate_rad_s"), values, strict=True))


def _turn(start: float, end: float) -> float:
    """The signed angle from one yaw to another, for yaw reported wrapped to (-pi, pi]."""
    return math.atan2(math.sin(end - start), math.cos(end - start))


def _mean_velocity(poses: Sequence[tuple[float, float, float, float]], window_s: float) -> tuple[float, float, float]:
    """Base velocity (forward, left, yaw rate) in the heading frame, averaged over the last `window_s`.

    Poses are (time, x, y, yaw). The average runs back to the newest pose at least `window_s` old, or to the
    first one while the travel is younger than that; a window of 0 measures the latest interval alone.
    """
    now = poses[-1]
    past = next((pose for pose in reversed(poses[:-1]) if now[0] - pose[0] >= window_s - 1e-9), poses[0])
    seconds = now[0] - past[0]
    if seconds <= 0:
        return (0.0, 0.0, 0.0)
    turn = sum(_turn(a[3], b[3]) for a, b in itertools.pairwise(poses[poses.index(past) :]))
    heading = past[3] + turn / 2  # the chord of a turning path points along its mean heading
    dx, dy = now[1] - past[1], now[2] - past[2]
    return (
        (dx * math.cos(heading) + dy * math.sin(heading)) / seconds,
        (dy * math.cos(heading) - dx * math.sin(heading)) / seconds,
        turn / seconds,
    )


# --------------------------------------------------------------------------
# The profiles: each one whole, looked up by name
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JointReference:
    """The controller's commanded joint state for one resource, verified against measurements before use."""

    positions: Mapping[str, float]
    velocities: Mapping[str, float]
    accelerations: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    """One measurement of a goal's first resource, as the runtime hands it to the goal's profile."""

    values: Mapping[str, Any]  # the resource's Sample.values
    time_s: float  # execution time of the measurement
    elapsed_s: float = 0.0  # since the goal was activated
    dt_s: float = 0.0  # since the previous reading
    still_for_s: float = 0.0  # how long the backend has measured the resource at rest under the fallback
    reference: JointReference | None = None  # controller reference for smooth mid-motion handoff


@dataclass(frozen=True)
class Profile[A: Model, L: Model, J: MotionJudge]:
    """One kind of motion, whole: the request it accepts, the limits a binding declares for it, how a request is
    admitted, and how its success is judged. Pass one to register() to add a profile; the runtime, the service,
    the client and `homunculus goal submit` look it up by name and need no other change."""

    name: str
    action: type[A]  # the request; its `profile` field is Literal[name]
    limits: type[L]  # what a binding declares; its `profile` field is Literal[name]
    resolve: Callable[[A, L], A]  # refuse what the limits do not allow; return the request with every setting concrete
    judge: Callable[[A, L, Reading], J]  # a judge for one activation, from the measurement at activation
    # Feed the judge one newer measurement. A backend that reports goal samples at its own cadence calls the
    # judge's update itself instead.
    measure: Callable[[J, Reading], ActivityStatus]
    activate: Callable[[A, L, Reading], A] = lambda action, limits, reading: action  # bind the request to the start
    duration_s: Callable[[A], float | None] = lambda action: None  # how long the request says its motion takes
    # Predicted sample values after this goal succeeds, for admission/validation of goals queued behind it.
    # Dispatch always re-checks real measurements.
    expected_end: Callable[[A, L, Reading], Mapping[str, Any] | None] = lambda action, limits, reading: None


PROFILES: dict[str, Profile[Any, Any, Any]] = {}


def register(profile: Profile[Any, Any, Any]) -> Profile[Any, Any, Any]:
    """Add a profile. Its name is what requests and bindings carry, so both models must say it, and no other
    profile may already use it."""
    for model in (profile.action, profile.limits):
        if get_args(model.model_fields["profile"].annotation) != (profile.name,):
            raise ValueError(f"{model.__name__}.profile must be Literal[{profile.name!r}]")
    if profile.name in PROFILES:
        raise ValueError(f"a profile named {profile.name!r} is already registered")
    PROFILES[profile.name] = profile
    return profile


def _measure_trajectory(judge: TrajectoryJudge, reading: Reading) -> ActivityStatus:
    joints = judge.action.joints
    positions = [reading.values["positions"][name] for name in joints]
    velocities = [reading.values["velocities"][name] for name in joints]
    return judge.update(reading.elapsed_s, positions, velocities, reading.dt_s)


def _base_pose(reading: Reading) -> Pose:
    positions = reading.values["positions"]
    return positions["x_m"], positions["y_m"], positions["yaw_rad"]


def _expected_trajectory_end(
    action: JointTrajectoryGoal, _limits: JointTrajectoryLimits, reading: Reading
) -> dict[str, Any]:
    last = action.points[-1]
    positions = {**reading.values["positions"], **dict(zip(action.joints, last.positions, strict=True))}
    velocities = {**reading.values["velocities"], **dict(zip(action.joints, last.velocities, strict=True))}
    return {**reading.values, "positions": positions, "velocities": velocities}


def _expected_posture_end(action: HandPostureGoal, limits: HandPostureLimits, reading: Reading) -> dict[str, Any]:
    posture = limits.postures[action.posture]
    return {
        **reading.values,
        "positions": {**reading.values["positions"], **posture},
        "velocities": {**reading.values["velocities"], **dict.fromkeys(posture, 0.0)},
    }


def _expected_pose_end(action: ToolPoseGoal, _limits: ToolPoseLimits, reading: Reading) -> dict[str, Any]:
    tool = {
        **reading.values.get("tool", {}),
        "position_m": list(action.position_m),
        "orientation_xyzw": list(action.orientation_xyzw),
    }
    return {**reading.values, "tool": tool}


register(
    Profile(
        name="joints.trajectory/1",
        action=JointTrajectoryGoal,
        limits=JointTrajectoryLimits,
        resolve=_resolve_trajectory,
        activate=lambda action, limits, reading: activate_trajectory(
            action, limits, reading.values["positions"], reading.values["velocities"], reference=reading.reference
        ),
        duration_s=lambda action: action.points[-1].time_s,
        expected_end=_expected_trajectory_end,
        judge=lambda action, limits, reading: TrajectoryJudge(action, limits),
        measure=_measure_trajectory,
    )
)
register(
    Profile(
        name="gripper.width/1",
        action=GripperWidthGoal,
        limits=GripperWidthLimits,
        resolve=_resolve_width,
        expected_end=lambda action, _limits, reading: {**reading.values, "width_m": action.width_m},
        judge=lambda action, limits, reading: WidthJudge(action, limits),
        measure=lambda judge, reading: judge.update(reading.values["width_m"], reading.dt_s),
    )
)
register(
    Profile(
        name="hand.posture/1",
        action=HandPostureGoal,
        limits=HandPostureLimits,
        resolve=_resolve_posture,
        expected_end=_expected_posture_end,
        judge=lambda action, limits, reading: PostureJudge(action, limits),
        measure=lambda judge, reading: judge.update(reading.values["positions"], reading.dt_s),
    )
)
register(
    Profile(
        name="tool.pose.linear/1",
        action=ToolPoseGoal,
        limits=ToolPoseLimits,
        resolve=_resolve_pose,
        expected_end=_expected_pose_end,
        judge=lambda action, limits, reading: PoseJudge(
            action, limits, reading.values["tool"]["position_m"], reading.values["tool"]["orientation_xyzw"]
        ),
        measure=lambda judge, reading: judge.update(
            reading.elapsed_s,
            reading.values["tool"]["position_m"],
            reading.values["tool"]["orientation_xyzw"],
            reading.dt_s,
        ),
    )
)
register(
    Profile(
        name="base.planar_velocity/1",
        action=BaseVelocityGoal,
        limits=BaseVelocityLimits,
        resolve=_resolve_velocity_goal,
        duration_s=lambda action: action.duration_s,
        judge=lambda action, limits, reading: VelocityJudge(
            action, limits, reading.time_s, _base_pose(reading), action.duration_s
        ),
        measure=lambda judge, reading: judge.update(reading.time_s, _base_pose(reading), reading.still_for_s),
    )
)
