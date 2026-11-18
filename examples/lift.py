"""A backend of your own, as small as it gets: one simulated linear lift that takes trajectory goals.

Serve it from this directory with:

    PYTHONPATH=. homunculus serve --backend lift:robots --backend-arg gain=15

Copy this file to start a backend for your robot. Keep describe() honest about what the
robot can do, send your robot its commands where _command() moves the simulated lift,
and read your robot's measurements where advance() reads the lift's. The runtime supplies
the judge that uses those measurements to decide when the goal is met.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from homunculus.models import (
    BackendInfo,
    ClockInfo,
    JointLimit,
    JointTrajectoryLimits,
    ProfileBinding,
    Sample,
    SensorInfo,
)
from homunculus.profiles import TrajectoryJudge, sample_trajectory
from homunculus.runtime import Backend, BackendFacts, Clock, Feedback, GoalCommand, ResourceSpec, SystemClock

LIMITS = JointTrajectoryLimits(
    profile="joints.trajectory/1",
    joints=[
        JointLimit(name="lift", unit="m", min_position=0.0, max_position=1.0, max_velocity=0.5, max_acceleration=2.0)
    ],
    max_points=16,
    max_duration_s=20.0,
    start_position_tolerance=0.01,
    start_velocity_tolerance=0.05,
    path_tolerance=0.05,
    goal_tolerance=0.005,
    settle_velocity=0.01,
    settle_time_s=0.2,
)


def robots(gain: str = "20") -> dict[str, Backend]:
    """What `--backend lift:robots` serves. Each --backend-arg arrives here as a string keyword argument."""
    return {"lift": Lift(SystemClock(), gain=float(gain))}


@dataclass
class _LiftGoal:
    id: str
    judge: TrajectoryJudge
    started_s: float


class Lift(Backend):
    reports_goal_samples = True

    def __init__(self, clock: Clock, gain: float):
        self.clock = clock
        self.gain = gain  # how hard the simulated motor pulls toward its reference, per second
        self.last = clock.now()
        self.time = 0.0  # execution time since the lift started
        self.position = self.velocity = self.reference = 0.0
        self.still_for = 0.0  # how long the lift has been measured standing still
        self.goal: _LiftGoal | None = None

    def describe(self) -> BackendFacts:
        return BackendFacts(
            info=BackendInfo(
                kind="lift",
                physics="simulated",  # "real" once this drives a physical robot
                notice="A simulated one-joint lift from examples/lift.py. It is not a physical robot.",
            ),
            world_id="lift-world",
            mode="position_control",
            clock_mode="realtime",
            clocks=[ClockInfo(id="lift_time", epoch="lift-world", description="seconds since the lift started")],
            frames=["world"],
            sensors=[
                SensorInfo(id="lift", kind="joint_state", provenance="simulation_ground_truth", description="height")
            ],
            revisions={"robot_description": "lift/1"},
            resources=[ResourceSpec(id="lift", role="lift", joints=["lift"], controller="lift_position")],
            bindings=[
                ProfileBinding(
                    profile="joints.trajectory/1",
                    target="lift",
                    role="lift",
                    resources=["lift"],
                    modes=["goal"],
                    controller="lift_position",
                    assistance="a first-order position servo follows the reference",
                    stop_deadline_wall_s=1.0,
                    limits=LIMITS,
                )
            ],
            concurrent=[],
            management=["recover"],
            period_s=0.01,
        )

    def apply(self, activity_id: str, contract: GoalCommand) -> None:
        self.goal = _LiftGoal(activity_id, contract.judge, self.time)

    def stop(self, resources: Sequence[str]) -> None:
        """The fallback: drop the goal and hold where the lift is measured to be."""
        self.goal, self.reference, self.still_for = None, self.position, 0.0

    def manage(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
        self.stop(parameters["resources"])  # recover, the one operation advertised: hold where it is now
        return {}

    def advance(self) -> Feedback:
        now = self.clock.now()
        dt, self.last = now - self.last, now
        if self.goal is not None:
            goal = self.goal
            (self.reference,), _ = sample_trajectory(goal.judge.action.points, self.time + dt - goal.started_s)
        self._command(dt)
        self.time += dt
        self.still_for = self.still_for + dt if abs(self.velocity) <= LIMITS.settle_velocity else 0.0

        feedback = Feedback(
            execution_time_s=self.time,
            samples={
                "lift": Sample(
                    source="lift",
                    provenance="simulation_ground_truth",
                    clock_id="lift_time",
                    clock_epoch="lift-world",
                    capture_time_ns=str(round(self.time * 1e9)),
                    valid=True,
                    values={"positions": {"lift": self.position}, "velocities": {"lift": self.velocity}},
                )
            },
            stopped={"lift": self.goal is None and self.still_for >= LIMITS.settle_time_s},
            holding={"lift": {"lift": self.reference}},
        )
        if self.goal is not None:
            goal = self.goal
            status = goal.judge.update(self.time - goal.started_s, [self.position], [self.velocity], dt)
            feedback.activities[goal.id] = status
            if status.phase in ("done", "failed"):  # reported once; the reference stays as the hold
                self.goal = None
        return feedback

    def _command(self, dt: float) -> None:
        """The simulated motor. On a real robot: send the reference, then read position and velocity back."""
        limit = LIMITS.joints[0].max_velocity
        self.velocity = max(-limit, min(limit, self.gain * (self.reference - self.position)))
        self.position += self.velocity * dt
