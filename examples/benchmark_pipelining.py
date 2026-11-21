"""Measurable before/after proof on real-time MuJoCo Panda physics.

Run:
    uv run python examples/benchmark_pipelining.py

Measures three claims on real-time wall-clock MuJoCo (100 Hz Franka Emika Panda):
  1. Arm idle fraction on a 4-step sequence with simulated agent think latency:
     stop-and-wait (before) vs causal goal queuing with `after_goal_id` (after).
  2. Mid-motion visual-servo redirect:
     `cancel_goal` + stop + restart (before) vs `replace_goal_id` + `joints.smooth/1` (after).
  3. Measurement-verified failure cascade:
     table collision fails `path_tolerance_violated` (never timer-based success) and
     auto-cancels the queued follow-up chain with zero dispatches.
"""

from __future__ import annotations

import itertools
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from homunculus import rigs
from homunculus.backends.mujoco import MujocoBackend
from homunculus.journal import Journal
from homunculus.models import (
    TERMINAL_GOAL_STATES,
    Authority,
    Goal,
    GoalSubmit,
    GripperWidthGoal,
    JointTrajectoryGoal,
    SessionOpen,
    TrajectoryPoint,
)
from homunculus.runtime import Principal, Runtime, SystemClock
from homunculus.smooth import JointSmoothGoal

CLIENT = Principal("agent")
WAYPOINTS = [
    {"joint1": 0.4, "joint4": -1.3},
    {"joint1": 0.8, "joint4": -1.8},
    {"joint1": -0.3, "joint4": -1.5},
    {"joint1": 0.0, "joint4": -1.57},
]
AGENT_THINK_S = 1.5


@dataclass
class _RealtimeLoop:
    runtime: Runtime
    condition: threading.Condition
    stop: threading.Event
    sampling: threading.Event
    total_ticks: int = 0
    idle_ticks: int = 0
    peak_joint1_torque: float = 0.0

    def wait_goal(self, goal_id: str) -> Goal:
        with self.condition:
            self.condition.wait_for(lambda: self.runtime.goal(goal_id).state in TERMINAL_GOAL_STATES)
        return self.runtime.goal(goal_id)


@contextmanager
def _realtime_panda(folder: Path, *, boot_id: str) -> Iterator[tuple[Runtime, _RealtimeLoop, str, str]]:
    """Run a real-time 100 Hz MuJoCo Panda runtime for the lifetime of a `with` block."""
    with (
        tempfile.TemporaryDirectory(prefix="homunculus-bench-") as tmp,
        Journal(Path(tmp) / "journal.sqlite", boot_id) as journal,
        MujocoBackend(SystemClock(), folder, smooth=True) as backend,
        Runtime("panda", backend, journal, boot_id=boot_id) as runtime,
    ):
        loop = _RealtimeLoop(
            runtime=runtime,
            condition=threading.Condition(),
            stop=threading.Event(),
            sampling=threading.Event(),
        )

        def run() -> None:
            due = time.monotonic()
            while not loop.stop.is_set():
                runtime.tick()
                if loop.sampling.is_set():
                    loop.total_ticks += 1
                    arm = next(r for r in runtime.describe().resources if r.id == "arm")
                    if arm.condition == "ready":
                        loop.idle_ticks += 1
                    torque = abs(runtime.observe().samples["arm"].values["actuator_forces"]["joint1"])
                    loop.peak_joint1_torque = max(loop.peak_joint1_torque, torque)
                with loop.condition:
                    loop.condition.notify_all()
                due += runtime.period_s
                remaining = due - time.monotonic()
                if remaining > 0:
                    loop.stop.wait(remaining)
                else:
                    due = time.monotonic()

        thread = threading.Thread(target=run, name=f"bench:{boot_id}")
        thread.start()
        try:
            context_id = runtime.describe().context_id
            session_id = runtime.open_session(
                CLIENT, SessionOpen(request_id=f"open-{boot_id}", context_id=context_id)
            ).session_id
            yield runtime, loop, context_id, session_id
        finally:
            loop.stop.set()
            thread.join(timeout=5.0)


def _submit(
    runtime: Runtime,
    requests: itertools.count[int],
    session_id: str,
    context_id: str,
    action,
    *,
    after_goal_id: str | None = None,
    replace_goal_id: str | None = None,
) -> Goal:
    return runtime.submit_goal(
        CLIENT,
        GoalSubmit(
            request_id=f"req-{next(requests)}",
            session_id=session_id,
            context_id=context_id,
            action=action,
            after_goal_id=after_goal_id,
            replace_goal_id=replace_goal_id,
        ),
    )


def benchmark_idle_fraction(folder: Path, *, pipeline: bool) -> tuple[float, float]:
    """Compare stop-and-wait vs causal queuing on a 4-step sequence with simulated agent think time."""
    requests = itertools.count(1)
    boot = "boot-after-queue" if pipeline else "boot-before-queue"
    with _realtime_panda(folder, boot_id=boot) as (runtime, loop, context_id, session_id):
        started = time.monotonic()
        loop.sampling.set()
        if not pipeline:
            for idx, target in enumerate(WAYPOINTS):
                goal = _submit(
                    runtime,
                    requests,
                    session_id,
                    context_id,
                    JointSmoothGoal(profile="joints.smooth/1", target="arm", positions=target, speed_scale=0.8),
                )
                done = loop.wait_goal(goal.goal_id)
                if done.state != "succeeded":
                    raise RuntimeError(f"step {idx} did not succeed: {done.state} ({done.result})")
                if idx < len(WAYPOINTS) - 1:
                    loop.stop.wait(AGENT_THINK_S)
        else:
            goals: list[Goal] = []
            for idx, target in enumerate(WAYPOINTS):
                after_id = goals[-1].goal_id if goals else None
                goals.append(
                    _submit(
                        runtime,
                        requests,
                        session_id,
                        context_id,
                        JointSmoothGoal(profile="joints.smooth/1", target="arm", positions=target, speed_scale=0.8),
                        after_goal_id=after_id,
                    )
                )
                # Agent thinks asynchronously while the arm executes instead of waiting for standstill.
                if idx == 0:
                    loop.stop.wait(0.2)
            for idx, goal in enumerate(goals):
                done = loop.wait_goal(goal.goal_id)
                if done.state != "succeeded":
                    raise RuntimeError(f"pipelined step {idx} did not succeed: {done.state} ({done.result})")
        loop.sampling.clear()
        elapsed_s = time.monotonic() - started
        idle_fraction = loop.idle_ticks / max(loop.total_ticks, 1)
        return idle_fraction, elapsed_s


def benchmark_redirect(folder: Path, *, smooth_replace: bool) -> tuple[float, float, float]:
    """Compare cancel-stop-restart vs mid-motion smooth replace during a visual-servo target correction."""
    requests = itertools.count(1)
    boot = "boot-after-replace" if smooth_replace else "boot-before-replace"
    initial = JointSmoothGoal(profile="joints.smooth/1", target="arm", positions={"joint1": 1.0}, speed_scale=0.8)
    redirect = JointSmoothGoal(profile="joints.smooth/1", target="arm", positions={"joint1": 1.5}, speed_scale=0.8)
    with _realtime_panda(folder, boot_id=boot) as (runtime, loop, context_id, session_id):
        g1 = _submit(runtime, requests, session_id, context_id, initial)
        loop.stop.wait(0.5)  # mid-motion snapshot happens while joint1 is moving fast
        decision_at = time.monotonic()
        loop.sampling.set()
        if not smooth_replace:
            runtime.cancel_goal(CLIENT, g1.goal_id, Authority(session_id=session_id, context_id=context_id))
            loop.wait_goal(g1.goal_id)
            stop_gap_s = time.monotonic() - decision_at
            g2 = _submit(runtime, requests, session_id, context_id, redirect)
        else:
            g2 = _submit(runtime, requests, session_id, context_id, redirect, replace_goal_id=g1.goal_id)
            stop_gap_s = 0.0
        done2 = loop.wait_goal(g2.goal_id)
        loop.sampling.clear()
        if done2.state != "succeeded":
            raise RuntimeError(f"redirect did not succeed: {done2.state} ({done2.result})")
        return stop_gap_s, time.monotonic() - decision_at, loop.peak_joint1_torque


def benchmark_collision_cascade(folder: Path) -> tuple[str, str, list[tuple[str, str, bool]]]:
    """Verify table collision fails by measurement (never a timer) and auto-cancels the queued tail."""
    requests = itertools.count(1)
    with _realtime_panda(folder, boot_id="boot-collision") as (runtime, loop, context_id, session_id):
        collide = JointTrajectoryGoal(
            profile="joints.trajectory/1",
            target="arm",
            joints=["joint2", "joint4"],
            points=[TrajectoryPoint(time_s=2.0, positions=[1.6, -0.8], velocities=[0.0, 0.0])],
        )
        g1 = _submit(runtime, requests, session_id, context_id, collide)
        g2 = _submit(
            runtime,
            requests,
            session_id,
            context_id,
            GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=0.02),
            after_goal_id=g1.goal_id,
        )
        g3 = _submit(
            runtime,
            requests,
            session_id,
            context_id,
            JointSmoothGoal(profile="joints.smooth/1", target="arm", positions={"joint2": 0.0, "joint4": -1.57}),
            after_goal_id=g2.goal_id,
        )
        done1 = loop.wait_goal(g1.goal_id)
        tails = []
        for tail in (g2, g3):
            done = loop.wait_goal(tail.goal_id)
            tails.append((done.state, done.result.reason if done.result else "", done.activated_time_ns is not None))
        return done1.state, done1.result.reason if done1.result else "", tails


def main() -> None:
    folder = rigs.find("panda")
    print("Running real-time 100 Hz MuJoCo Panda benchmark...")

    before_idle, before_wall = benchmark_idle_fraction(folder, pipeline=False)
    after_idle, after_wall = benchmark_idle_fraction(folder, pipeline=True)

    cancel_gap, cancel_wall, cancel_torque = benchmark_redirect(folder, smooth_replace=False)
    replace_gap, replace_wall, replace_torque = benchmark_redirect(folder, smooth_replace=True)

    col_state, col_reason, col_tails = benchmark_collision_cascade(folder)

    print("\n1. Multi-step manipulation sequence (4 moves, 1.5 s simulated agent think latency):")
    print(f"   Before (stop-and-wait):    idle = {before_idle * 100:5.1f}%, wall = {before_wall:5.2f} s")
    print(f"   After  (after_goal_id):    idle = {after_idle * 100:5.1f}%, wall = {after_wall:5.2f} s")

    print("\n2. Mid-motion visual-servo redirect (joint1 1.0 -> 1.5 rad at t = 0.5 s):")
    print(
        f"   Before (cancel + restart): stop gap = {cancel_gap * 1000:5.0f} ms, "
        f"arrival = {cancel_wall:4.2f} s, peak torque = {cancel_torque:5.2f} N*m"
    )
    print(
        f"   After  (replace_goal_id):  stop gap = {replace_gap * 1000:5.0f} ms, "
        f"arrival = {replace_wall:4.2f} s, peak torque = {replace_torque:5.2f} N*m"
    )

    print("\n3. Measurement-verified failure cascade (table collision -> queued gripper -> queued lift):")
    print(f"   Colliding move: {col_state} ({col_reason})")
    for idx, (state, reason, dispatched) in enumerate(col_tails, start=2):
        print(f"   Queued step {idx}:  {state} ({reason}), dispatched={dispatched}")


if __name__ == "__main__":
    main()
