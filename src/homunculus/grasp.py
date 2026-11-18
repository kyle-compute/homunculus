"""gripper.grasp/1: close a parallel-jaw gripper on what is between its jaws, and say whether it holds something.

Defined outside the core modules, as any profile can be (see profiles.Profile): importing this module registers it.
A backend that binds it closes the jaws when it applies a goal, feeds the judge the measured gap, and keeps the
jaws squeezing after the goal ends, so what they hold stays held until another goal opens them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .models import Model, RobotError
from .profiles import ActivityStatus, MotionJudge, Profile, measures, register


class GripperGraspGoal(Model):
    """Close the jaws until they stop. The grasp holds something if they stop wider than min_width_m and no wider
    than max_width_m, the size of what they hold; left out, those are the binding's empty_width_m and max_width_m."""

    profile: Literal["gripper.grasp/1"]
    target: str
    min_width_m: float | None = Field(default=None, ge=0)
    max_width_m: float | None = Field(default=None, gt=0)


class GripperGraspLimits(Model):
    profile: Literal["gripper.grasp/1"]
    max_width_m: float  # fully open
    empty_width_m: float  # closed on nothing, the jaws measure at most this gap
    still_tolerance_m: float  # how far the gap may move while the jaws count as stopped
    settle_time_s: float  # how long they stay stopped before the verdict


class GraspJudge(MotionJudge):
    """Waits until the gap has stayed within still_tolerance_m of one value for settle_time_s, then judges that
    gap against the size the request expects."""

    def __init__(self, action: GripperGraspGoal, limits: GripperGraspLimits) -> None:
        super().__init__()
        self.action, self.limits = action, limits
        self.stopped_at_m: float | None = None
        self.still_for = 0.0

    @measures
    def update(self, width_m: float, dt: float) -> ActivityStatus:
        if self.stopped_at_m is None or abs(width_m - self.stopped_at_m) > self.limits.still_tolerance_m:
            self.stopped_at_m, self.still_for = width_m, 0.0
            return ActivityStatus("executing")
        self.still_for += dt
        if self.still_for < self.limits.settle_time_s:
            return ActivityStatus("finishing")
        low, high = self.action.min_width_m, self.action.max_width_m  # concrete since _resolve_grasp
        measurements = {"width_m": width_m}
        if width_m <= low:
            detail = f"the jaws closed to {width_m * 100:.1f} cm, no wider than {low * 100:.1f} cm: they hold nothing"
            return ActivityStatus("failed", measurements, error="nothing_grasped", detail=detail)
        if width_m > high:
            detail = f"the jaws stopped at {width_m * 100:.1f} cm, wider than the {high * 100:.1f} cm asked for"
            return ActivityStatus("failed", measurements, error="unexpected_width", detail=detail)
        return ActivityStatus("done", measurements)


def _resolve_grasp(action: GripperGraspGoal, limits: GripperGraspLimits) -> GripperGraspGoal:
    low = limits.empty_width_m if action.min_width_m is None else action.min_width_m
    high = limits.max_width_m if action.max_width_m is None else action.max_width_m
    if not low < high <= limits.max_width_m:
        raise RobotError(
            "limit_exceeded",
            f"a grasp needs min_width_m < max_width_m <= {limits.max_width_m:g}; got {low:g} and {high:g}",
        )
    return action.model_copy(update={"min_width_m": low, "max_width_m": high})


register(
    Profile(
        name="gripper.grasp/1",
        action=GripperGraspGoal,
        limits=GripperGraspLimits,
        resolve=_resolve_grasp,
        judge=lambda action, limits, reading: GraspJudge(action, limits),
        measure=lambda judge, reading: judge.update(reading.values["width_m"], reading.dt_s),
    )
)
