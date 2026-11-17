"""tool.pose.linear/1: what admission refuses, and how the judge decides from measured tool poses."""

from __future__ import annotations

import math

import pytest

from homunculus.backends.task import POSE_LIMITS
from homunculus.models import RobotError, ToolPoseGoal, ToolPoseLimits
from homunculus.profiles import PoseJudge, _resolve_pose, quaternion_angle

LIMITS = ToolPoseLimits(**POSE_LIMITS, control_point="left_tool")
LEVEL = [0.0, 0.0, 0.0, 1.0]
QUARTER_TURN_Z = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]


def goal(**overrides) -> ToolPoseGoal:
    fields = {"profile": "tool.pose.linear/1", "target": "left_hand", "position_m": [0.4, 0.2, 0.9]}
    return ToolPoseGoal(**{"orientation_xyzw": LEVEL, **fields, **overrides})


@pytest.mark.parametrize(
    ("overrides", "code", "parameter"),
    [
        ({"orientation_xyzw": [0, 0, 0, 2]}, "invalid_request", "orientation_xyzw"),
        ({"position_m": [0.4, 0.2, 0.4]}, "limit_exceeded", "position_m.z"),
        ({"speed_m_s": 0.5}, "limit_exceeded", "speed_m_s"),
        ({"frame_id": "base_heading"}, "unsupported_parameter", "frame_id"),
        ({"position_tolerance_m": 0.2}, "limit_exceeded", "position_tolerance_m"),
    ],
)
def test_admission_refuses_what_the_binding_does_not_allow(overrides, code, parameter):
    with pytest.raises(RobotError) as refused:
        _resolve_pose(goal(**overrides), LIMITS)
    assert refused.value.code == code
    assert refused.value.details["parameter"] == parameter


def test_resolution_states_every_setting():
    resolved = _resolve_pose(goal(orientation_xyzw=[0, 0, 0, 1.0004]), LIMITS)
    assert resolved.speed_m_s == LIMITS.default_speed_m_s
    assert resolved.position_tolerance_m == LIMITS.position_tolerance_m
    assert resolved.orientation_tolerance_rad == LIMITS.orientation_tolerance_rad
    assert math.isclose(math.sqrt(sum(v * v for v in resolved.orientation_xyzw)), 1.0)


def run(judge: PoseJudge, measure, dt: float = 0.1, limit_s: float = 20.0):
    elapsed, status = 0.0, None
    while elapsed < limit_s:
        elapsed += dt
        position, xyzw = measure(elapsed)
        status = judge.update(elapsed, position, xyzw, dt)
        if status.phase in ("done", "failed"):
            return elapsed, status
    raise AssertionError(f"undecided after {limit_s} s: {status}")


def test_a_hand_that_follows_the_reference_succeeds_after_settling():
    action = _resolve_pose(goal(orientation_xyzw=QUARTER_TURN_Z), LIMITS)
    judge = PoseJudge(action, LIMITS, [0.4, 0.0, 0.9], LEVEL)
    # 0.2 m at 0.08 m/s peak is 3.75 s; a quarter turn at 0.5 rad/s peak is 4.7 s, so rotation sets the time.
    assert math.isclose(judge.duration_s, 1.5 * (math.pi / 2) / 0.5)
    assert math.isclose(quaternion_angle(judge.reference(judge.duration_s / 2)[1], LEVEL), math.pi / 4, rel_tol=1e-6)
    elapsed, status = run(judge, judge.reference)
    assert status.phase == "done"
    assert elapsed >= judge.duration_s + LIMITS.settle_time_s
    assert status.measurements["position_error_m"] < 1e-9


def test_a_hand_pushed_off_the_segment_fails_while_moving():
    action = _resolve_pose(goal(), LIMITS)
    judge = PoseJudge(action, LIMITS, [0.4, 0.0, 0.9], LEVEL)

    def pushed(elapsed):
        position, xyzw = judge.reference(elapsed)
        return [position[0], position[1], position[2] + (0.08 if elapsed > 1.0 else 0.0)], xyzw

    elapsed, status = run(judge, pushed)
    assert status.error == "path_tolerance_violated"
    assert 1.0 < elapsed <= judge.duration_s


def test_a_hand_that_stops_short_fails_after_the_settle_wait():
    action = _resolve_pose(goal(), LIMITS)
    judge = PoseJudge(action, LIMITS, [0.4, 0.0, 0.9], LEVEL)

    def short(elapsed):  # on the segment, but stuck 3 cm before its end
        position, xyzw = judge.reference(min(elapsed, judge.duration_s))
        return [position[0], min(position[1], 0.17), position[2]], xyzw

    elapsed, status = run(judge, short)
    assert status.error == "goal_not_reached"
    assert math.isclose(status.measurements["position_error_m"], 0.03, abs_tol=1e-9)
    assert elapsed >= judge.duration_s + LIMITS.max_settle_wait_s
