"""joints.smooth/1: auto-timed C2 quintic joint motion from rest or mid-flight.

Defined outside the core modules, as any profile can be (see profiles.Profile): importing this module registers it.
Given target joint positions, the profile computes the shortest synchronized C2 quintic duration within the binding's
position, velocity, and acceleration limits — from rest or directly from a verified mid-motion reference — while still
judging completion strictly from measured joint positions and velocities.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from .models import JointLimit, Model, RobotError
from .profiles import ActivityStatus, MotionJudge, Profile, Reading, measures, register, verified_joint_start


class JointSmoothGoal(Model):
    """Move named joints smoothly to target positions without guessing a trajectory duration."""

    profile: Literal["joints.smooth/1"]
    target: str
    positions: dict[str, float] = Field(min_length=1)
    speed_scale: float = Field(default=1.0, gt=0.0, le=1.0)
    min_duration_s: float | None = Field(default=None, ge=0.0)
    path_tolerance: float | None = Field(default=None, gt=0.0)
    goal_tolerance: float | None = Field(default=None, gt=0.0)


class JointSmoothLimits(Model):
    profile: Literal["joints.smooth/1"]
    joints: list[JointLimit]
    max_duration_s: float
    start_position_tolerance: float
    path_tolerance: float
    goal_tolerance: float
    settle_velocity: float
    settle_time_s: float


@dataclass(frozen=True)
class _Quintic:
    """p(s) = c0 + c1*s + c2*s^2 + c3*s^3 + c4*s^4 + c5*s^5 on s = t / duration in [0, 1]."""

    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    target: float

    @classmethod
    def fit(cls, q0: float, v0: float, a0: float, q1: float, duration: float) -> _Quintic:
        delta = q1 - q0
        c1 = v0 * duration
        c2 = 0.5 * a0 * duration * duration
        c3 = 10.0 * delta - 6.0 * c1 - 3.0 * c2
        c4 = -15.0 * delta + 8.0 * c1 + 3.0 * c2
        c5 = 6.0 * delta - 3.0 * c1 - c2
        return cls(q0, c1, c2, c3, c4, c5, q1)

    def position(self, s: float) -> float:
        return (((((self.c5 * s + self.c4) * s + self.c3) * s + self.c2) * s + self.c1) * s) + self.c0

    def dp(self, s: float) -> float:
        return (((5.0 * self.c5 * s + 4.0 * self.c4) * s + 3.0 * self.c3) * s + 2.0 * self.c2) * s + self.c1

    def ddp(self, s: float) -> float:
        return ((20.0 * self.c5 * s + 12.0 * self.c4) * s + 6.0 * self.c3) * s + 2.0 * self.c2


def _bisect_roots(fn: Callable[[float], float], knots: Sequence[float]) -> list[float]:
    """Find roots of a polynomial inside (0, 1) given strictly monotone brackets from its derivative's roots."""
    roots: list[float] = []
    for lo, hi in itertools.pairwise(knots):
        f_lo, f_hi = fn(lo), fn(hi)
        if f_lo == 0.0 and 0.0 < lo < 1.0:
            roots.append(lo)
            continue
        if f_lo * f_hi >= 0.0:
            continue
        a, b, fa = lo, hi, f_lo
        for _ in range(32):
            mid = 0.5 * (a + b)
            fm = fn(mid)
            if fa * fm <= 0.0:
                b = mid
            else:
                a, fa = mid, fm
        roots.append(0.5 * (a + b))
    return roots


def _quintic_extrema(poly: _Quintic, duration: float) -> tuple[float, float, float, float]:
    """Exact (min_q, max_q, peak_speed, peak_acceleration) on s in [0, 1] via Rolle's theorem."""
    # p'''(s) = 6 * (c3 + 4*c4*s + 10*c5*s^2) = 0 gives exact critical points of acceleration p''(s).
    a, b, c = 10.0 * poly.c5, 4.0 * poly.c4, poly.c3
    dddp_roots: list[float] = []
    if abs(a) <= 1e-12:
        if abs(b) > 1e-12:
            r = -c / b
            if 0.0 < r < 1.0:
                dddp_roots.append(r)
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            root = math.sqrt(disc)
            dddp_roots = sorted(r for r in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if 0.0 < r < 1.0)

    acc_pts = [0.0, *dddp_roots, 1.0]
    inv_t = 1.0 / duration
    inv_t2 = inv_t * inv_t
    peak_acc = max(abs(poly.ddp(s)) * inv_t2 for s in acc_pts)

    ddp_roots = _bisect_roots(poly.ddp, acc_pts)
    vel_pts = [0.0, *ddp_roots, 1.0]
    peak_speed = max(abs(poly.dp(s)) * inv_t for s in vel_pts)

    dp_roots = _bisect_roots(poly.dp, vel_pts)
    pos_vals = [poly.position(s) for s in (0.0, *dp_roots, 1.0)]
    return min(pos_vals), max(pos_vals), peak_speed, peak_acc


@dataclass(frozen=True)
class SmoothPlan:
    """One solved synchronized C2 quintic motion across a goal's joints."""

    joints: tuple[str, ...]
    duration_s: float
    curves: tuple[_Quintic, ...]

    def sample(self, elapsed: float) -> tuple[list[float], list[float]]:
        if self.duration_s <= 0.0 or elapsed >= self.duration_s:
            return [curve.target for curve in self.curves], [0.0] * len(self.curves)
        s = max(0.0, elapsed) / self.duration_s
        inv_t = 1.0 / self.duration_s
        return [curve.position(s) for curve in self.curves], [curve.dp(s) * inv_t for curve in self.curves]


def solve_smooth(
    action: JointSmoothGoal,
    limits: JointSmoothLimits,
    reading: Reading,
) -> SmoothPlan:
    """Compute the shortest synchronized C2 quintic within position, velocity, and acceleration limits."""
    joints = tuple(action.positions)
    known = {joint.name: joint for joint in limits.joints}
    joint_limits = [known[name] for name in joints]
    targets = [action.positions[name] for name in joints]
    start_q, start_v, raw_a = verified_joint_start(
        joints,
        reading.values["positions"],
        reading.values["velocities"],
        path_tolerance=action.path_tolerance or limits.path_tolerance,
        settle_velocity=limits.settle_velocity,
        reference=reading.reference,
    )
    # Clamp finite-difference start acceleration inside each joint's limit so boundary noise cannot force infeasibility.
    start_a = [
        max(-0.95 * j.max_acceleration, min(0.95 * j.max_acceleration, a))
        for j, a in zip(joint_limits, raw_a, strict=True)
    ]

    if all(
        abs(q1 - q0) <= 1e-12 and abs(v0) <= 1e-12 and abs(a0) <= 1e-12
        for q0, v0, a0, q1 in zip(start_q, start_v, start_a, targets, strict=True)
    ):
        duration = action.min_duration_s or 0.0
        curves = tuple(_Quintic(q1, 0.0, 0.0, 0.0, 0.0, 0.0, q1) for q1 in targets)
        return SmoothPlan(joints=joints, duration_s=duration, curves=curves)

    slack = limits.start_position_tolerance
    eps = 1e-9

    def build_if_feasible(duration: float) -> tuple[_Quintic, ...] | None:
        curves: list[_Quintic] = []
        for j, q0, v0, a0, q1 in zip(joint_limits, start_q, start_v, start_a, targets, strict=True):
            poly = _Quintic.fit(q0, v0, a0, q1, duration)
            low, high, speed, acc = _quintic_extrema(poly, duration)
            if (
                low < j.min_position - slack - eps
                or high > j.max_position + slack + eps
                or speed > j.max_velocity * action.speed_scale + eps
                or acc > j.max_acceleration + eps
            ):
                return None
            curves.append(poly)
        return tuple(curves)

    lower = max(
        action.min_duration_s or 0.0,
        1e-3,
        max(
            abs(q1 - q0) / (j.max_velocity * action.speed_scale)
            for j, q0, q1 in zip(joint_limits, start_q, targets, strict=True)
        ),
        max(abs(v0) / j.max_acceleration for j, v0 in zip(joint_limits, start_v, strict=True)),
    )
    if lower > limits.max_duration_s:
        raise RobotError(
            "limit_exceeded",
            f"smooth motion needs at least {lower:.3f} s, exceeding max_duration_s={limits.max_duration_s:g}",
            parameter="max_duration_s",
            minimum=lower,
            maximum=limits.max_duration_s,
        )

    # Scan geometrically first: heading toward a limit at nonzero speed makes feasibility a window [t_lo, t_hi].
    bad = lower
    good: float | None = None
    best_curves: tuple[_Quintic, ...] | None = None
    candidate = lower
    while True:
        curves = build_if_feasible(candidate)
        if curves is not None:
            good, best_curves = candidate, curves
            break
        bad = candidate
        if candidate >= limits.max_duration_s:
            break
        candidate = min(limits.max_duration_s, candidate * 1.15)

    if good is None or best_curves is None:
        raise RobotError(
            "limit_exceeded",
            f"no feasible C2 smooth motion within max_duration_s={limits.max_duration_s:g} respects joint limits",
            parameter="max_duration_s",
            maximum=limits.max_duration_s,
        )

    if good > lower:
        lo, hi = bad, good
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            curves = build_if_feasible(mid)
            if curves is not None:
                hi, best_curves = mid, curves
            else:
                lo = mid
        good = hi

    return SmoothPlan(joints=joints, duration_s=good, curves=best_curves)


class SmoothJudge(MotionJudge):
    """joints.smooth/1. Follows the solved C2 quintic and judges arrival strictly from measurements."""

    def __init__(self, action: JointSmoothGoal, limits: JointSmoothLimits, plan: SmoothPlan) -> None:
        super().__init__()
        self.action, self.limits, self.plan = action, limits, plan
        self.settled_for = 0.0
        self.peak_error = 0.0

    @property
    def joints(self) -> Sequence[str]:
        return self.plan.joints

    def reference(self, elapsed: float) -> tuple[list[float], list[float]]:
        return self.plan.sample(elapsed)

    def end_positions(self) -> dict[str, float]:
        return dict(self.action.positions)

    @measures
    def update(
        self, elapsed: float, positions: Sequence[float], velocities: Sequence[float], dt: float
    ) -> ActivityStatus:
        action, limits = self.action, self.limits
        reference, _ = self.plan.sample(elapsed)
        errors = [abs(q - r) for q, r in zip(positions, reference, strict=True)]
        worst = max(range(len(errors)), key=errors.__getitem__)
        self.peak_error = max(self.peak_error, errors[worst])
        if errors[worst] > action.path_tolerance:
            measurements = {
                "joint": self.plan.joints[worst],
                "tracking_error": errors[worst],
                "path_tolerance": action.path_tolerance,
                "measured": positions[worst],
                "reference": reference[worst],
            }
            detail = (
                f"{self.plan.joints[worst]} trailed the reference by {errors[worst]:.4f}; "
                f"the limit is {action.path_tolerance:g}"
            )
            return ActivityStatus("failed", measurements, error="path_tolerance_violated", detail=detail)
        if elapsed < self.plan.duration_s:
            return ActivityStatus("executing")
        goal_errors = [abs(q - curve.target) for q, curve in zip(positions, self.plan.curves, strict=True)]
        settled = max(goal_errors) <= action.goal_tolerance and max(map(abs, velocities)) <= limits.settle_velocity
        self.settled_for = self.settled_for + dt if settled else 0.0
        if self.settled_for < limits.settle_time_s:
            return ActivityStatus("finishing")
        return ActivityStatus(
            "done",
            {
                "duration_s": self.plan.duration_s,
                "peak_tracking_error": self.peak_error,
                "final_error": dict(zip(self.plan.joints, goal_errors, strict=True)),
            },
        )


def _resolve_smooth(action: JointSmoothGoal, limits: JointSmoothLimits) -> JointSmoothGoal:
    known = {joint.name: joint for joint in limits.joints}
    unknown = [name for name in action.positions if name not in known]
    if unknown:
        raise RobotError(
            "unsupported_parameter",
            f"{action.target!r} has no joint named {', '.join(map(repr, unknown))}; it has {', '.join(known)}",
            parameter="positions",
            supported=sorted(known),
        )
    for name, target in action.positions.items():
        joint = known[name]
        if target < joint.min_position or target > joint.max_position:
            raise RobotError(
                "limit_exceeded",
                f"{name} position={target:g} is outside the supported range "
                f"{joint.min_position:g} to {joint.max_position:g}",
                parameter=f"{name} position",
                value=target,
                minimum=joint.min_position,
                maximum=joint.max_position,
            )
    if action.min_duration_s is not None and action.min_duration_s > limits.max_duration_s:
        raise RobotError(
            "limit_exceeded",
            f"min_duration_s={action.min_duration_s:g} exceeds max_duration_s={limits.max_duration_s:g}",
            parameter="min_duration_s",
            value=action.min_duration_s,
            maximum=limits.max_duration_s,
        )
    tolerances: dict[str, float] = {}
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
    return action.model_copy(update=tolerances)


def _activate_smooth(action: JointSmoothGoal, limits: JointSmoothLimits, reading: Reading) -> JointSmoothGoal:
    solve_smooth(action, limits, reading)
    return action


def _expected_smooth_end(action: JointSmoothGoal, _limits: JointSmoothLimits, reading: Reading) -> dict[str, Any]:
    return {
        **reading.values,
        "positions": {**reading.values["positions"], **action.positions},
        "velocities": {**reading.values["velocities"], **dict.fromkeys(action.positions, 0.0)},
    }


def _measure_smooth(judge: SmoothJudge, reading: Reading) -> ActivityStatus:
    positions = [reading.values["positions"][name] for name in judge.joints]
    velocities = [reading.values["velocities"][name] for name in judge.joints]
    return judge.update(reading.elapsed_s, positions, velocities, reading.dt_s)


register(
    Profile(
        name="joints.smooth/1",
        action=JointSmoothGoal,
        limits=JointSmoothLimits,
        resolve=_resolve_smooth,
        activate=_activate_smooth,
        expected_end=_expected_smooth_end,
        judge=lambda action, limits, reading: SmoothJudge(action, limits, solve_smooth(action, limits, reading)),
        measure=_measure_smooth,
    )
)
