"""The `homunculus` command: arguments, client calls, and terminal or JSON presentation.

Every robot command goes through RobotClient. Results print as readable text,
or as one JSON document with --json; errors and hints go to stderr.

Exit status: 0 success, 1 the robot operation did not succeed, 2 bad usage or
input, 3 the service refused the request, 4 no target or service to talk to,
130 interrupted, 141 the reader closed the output.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import itertools
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, NoReturn

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from . import __version__, journal, rigs
from .client import STOP_WITHOUT_SESSION, ClientError, RequestInterrupted, RobotClient, TargetStore, WaitTimeout
from .models import (
    PROTOCOL,
    TERMINAL_GOAL_STATES,
    Action,
    BaseVelocityGoal,
    BaseVelocityTarget,
    Goal,
    GoalSpec,
    GoalSubmit,
    GripperWidthGoal,
    HandPostureGoal,
    JointTrajectoryGoal,
    Observation,
    Operation,
    ProfileBinding,
    RobotDescription,
    RobotError,
    Session,
    StartConditions,
    Stream,
    StreamOpen,
    StreamSpec,
    StreamUpdate,
    ToolPoseGoal,
    TrajectoryPoint,
)
from .runtime import StartupError

OK, FAILED, USAGE, REFUSED, NO_SERVICE, INTERRUPTED, OUTPUT_CLOSED = 0, 1, 2, 3, 4, 130, 141

HINTS = {
    "stale_context": "the robot restarted, was reset, or was recovered; run `homunculus session open` to continue",
    "session_invalid": "open a new session with `homunculus session open`",
    "session_conflict": (
        "if that session is yours, keep using it (`homunculus session renew` extends it, `homunculus session close` "
        "releases it); otherwise wait for it to close or expire"
    ),
    "journal_full": "restart the service: that retires this execution context, and its finished requests make room",
    "resource_faulted": "an operator must recover the robot: `homunculus manage recover`",
    "resource_busy": (
        "wait for the current activity, end it with `homunculus goal cancel` or `homunculus stream close`, "
        "or stop everything with `homunculus stop`"
    ),
    "stream_closed": "open a new stream with `homunculus stream open`",
    "target_expired": (
        "base the target on a newer observation: read one with `homunculus observe` and name it with --basis, "
        "or leave --basis out to use one read now"
    ),
    "basis_invalid": (
        "name an observation read in this context, and not before the one the previous target was based on; "
        "or leave --basis out to use one read now"
    ),
    "unauthenticated": (
        "the service did not accept this token; connect again with a current one: "
        "`homunculus connect URL --token TOKEN`, or `homunculus connect --home DIR` on the service's machine"
    ),
    "unknown_robot": "list this service's robots with `homunculus targets`",
}


# The command that shows the operation a request ID belongs to, by the kind of request.
INSPECT = {"goal": "goal inspect", "stream": "stream inspect", "management": "operation inspect"}
# The flag that sets each goal deadline.
DEADLINE_FLAGS = {"max_execution_s": "--max-execution", "max_wall_s": "--max-wall"}


class UsageError(Exception):
    """Input this command cannot use; reported with exit status 2."""


class WaitInterrupted(Exception):
    """Ctrl-C while a command waited for the robot; reported with exit status 130."""

    def __init__(self, waited_for: str, advice: str):
        super().__init__(waited_for)
        self.message = f"interrupted while waiting for {waited_for}"
        self.hint = f"only the wait ended: {advice}"


@contextmanager
def _waiting(waited_for: str, advice: str) -> Iterator[None]:
    """A wait for the robot. Ctrl-C ends the wait and nothing else, so the command says what goes on without it."""
    try:
        yield
    except KeyboardInterrupt:
        raise WaitInterrupted(waited_for, advice) from None


GOAL_GOES_ON = (
    "the goal goes on; see it with `homunculus goal inspect`, end it with `homunculus goal cancel` or `homunculus stop`"
)


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def _emit(args: argparse.Namespace, document: Any, text: str) -> None:
    """Print one result: a JSON document for automation, or text for a person."""
    if args.json:
        payload = document.model_dump(mode="json") if isinstance(document, BaseModel) else document
        print(json.dumps(payload, indent=2))
    else:
        print(text)


def _width() -> int:
    """Columns to wrap text to: the terminal's, or 100 when the output is not a terminal."""
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 120))


def _columns(rows: list[tuple[str, ...]], indent: int = 2) -> str:
    """Rows as aligned columns. The last column wraps under itself instead of running past the terminal."""
    widths = [max(map(len, column)) for column in zip(*rows, strict=True)]
    lead = indent + sum(widths[:-1]) + 2 * (len(widths) - 1)
    room = max(30, _width() - lead)
    lines = []
    for row in rows:
        head = " " * indent + "".join(cell.ljust(width) + "  " for cell, width in zip(row[:-1], widths, strict=False))
        tail = textwrap.wrap(row[-1], room, break_on_hyphens=False) or [""]
        lines.append((head + tail[0]).rstrip())
        lines.extend(" " * lead + more for more in tail[1:])
    return "\n".join(lines)


def _flat(value: Any, nested: bool = False) -> str:
    if isinstance(value, Mapping):
        text = ", ".join(f"{key}={_flat(item, True)}" for key, item in value.items())
        return f"{{{text}}}" if nested else text or "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flat(item, True) for item in value) + "]"
    return str(value)


def _limit_rows(binding: ProfileBinding) -> list[tuple[str, str]]:
    """What a binding accepts, one limit per row. A profile with no layout here, such as one a fork adds, shows
    every limit as its binding declares it."""
    limits = binding.limits
    if limits.profile == "base.planar_velocity/1":
        rows = [
            ("command", f"velocity of {limits.control_point!r} in {', '.join(limits.frames)}"),
            ("forward", f"up to {limits.max_forward_m_s:g} m/s"),
            ("left", f"up to {limits.max_left_m_s:g} m/s"),
            ("yaw", f"up to {limits.max_yaw_rate_rad_s:g} rad/s"),
        ]
        if limits.min_update_validity_wall_s is not None:
            validity = f"{limits.min_update_validity_wall_s:g} to {limits.max_update_validity_wall_s:g} s"
            rows.append(("stream", f"each target valid for {validity}"))
        rows.append(
            (
                "timed goal",
                f"{limits.transient_s:g} to {limits.max_goal_duration_s:g} s, faster than "
                f"{limits.max_tracking_error_m_s:g} m/s or {limits.max_tracking_error_rad_s:g} rad/s "
                "(slower or shorter travel cannot be verified)",
            )
        )
        return rows
    if limits.profile == "joints.trajectory/1":
        rows = [(j.name, f"{j.min_position:g} to {j.max_position:g} {j.unit}") for j in limits.joints]
        return [*rows, ("trajectory", f"up to {limits.max_points} waypoints and {limits.max_duration_s:g} s")]
    if limits.profile == "hand.posture/1":
        return [("postures", ", ".join(sorted(limits.postures)))]
    if limits.profile == "gripper.width/1":
        return [
            ("width", f"{limits.min_width_m:g} to {limits.max_width_m:g} m"),
            ("completes on", ", ".join(limits.complete_on)),
        ]
    if limits.profile == "tool.pose.linear/1":
        low, high = limits.workspace_min_m, limits.workspace_max_m
        return [
            ("command", f"pose of {limits.control_point!r} in {', '.join(limits.frames)}, straight line"),
            ("workspace", ", ".join(f"{a} {lo:g} to {hi:g} m" for a, lo, hi in zip("xyz", low, high, strict=True))),
            (
                "speed",
                f"{limits.min_speed_m_s:g} to {limits.max_speed_m_s:g} m/s (default {limits.default_speed_m_s:g})",
            ),
            (
                "reached",
                f"within {limits.position_tolerance_m:g} m and {limits.orientation_tolerance_rad:g} rad by default; "
                f"a goal may allow up to {limits.max_position_tolerance_m:g} m and "
                f"{limits.max_orientation_tolerance_rad:g} rad",
            ),
        ]
    return [(name, _flat(value)) for name, value in limits.model_dump(exclude={"profile"}).items()]


def _description_text(d: RobotDescription) -> str:
    loop, limits = d.control_loop, d.limits
    session = f"held by {d.session.principal}, expires in {d.session.expires_in_s:.0f} s" if d.session.held else "free"
    timing = f"{loop.ticks} ticks, longest interval {loop.worst_interval_s:.3g} s"
    if loop.execution_lag_s is not None:
        timing += f", execution lag {loop.execution_lag_s:.3g} s"

    def cameras(kind: str) -> str:
        found = [
            s.id + (f" (live video, {s.video_fps:g} fps)" if s.video_fps else "") for s in d.sensors if s.kind == kind
        ]
        return ", ".join(found) or "none"

    physics = {"none": "no physics", "simulated": "simulated physics", "real": "a physical robot"}[d.backend.physics]
    lines = [
        f"{d.robot_id}  ({d.backend.kind} backend, {physics})",
        textwrap.fill(d.backend.notice, _width(), initial_indent="  ", subsequent_indent="  "),
        "",
        _columns(
            [
                ("context", d.context_id),
                ("world", f"{d.world_id}  (boot {d.boot_id})"),
                ("mode", f"{d.mode}, {d.clock_mode} clock, control period {loop.period_s:g} s"),
                ("timing", timing),
                ("session", session),
                ("frames", ", ".join(d.frames)),
                ("cameras", cameras("camera")),
                ("spectators", f"{cameras('spectator_camera')}  (for people; never in observations)"),
                ("management", ", ".join(d.management) or "none"),
            ]
        ),
        "",
        "profiles",
    ]
    for binding in d.profiles:
        state = "" if binding.available else f"  UNAVAILABLE: {binding.unavailable_reason}"
        lines.append(f"  {binding.profile} on {binding.target!r} ({' and '.join(binding.modes)}){state}")
        rows = [*_limit_rows(binding), ("controller", binding.controller), ("assistance", binding.assistance)]
        lines.append(_columns(rows, indent=4))
    lines += ["", "resources"]
    rows = []
    for resource in d.resources:
        using = f"used by {resource.activity}" if resource.activity else "idle"
        held = (
            f"holds {resource.retained.kind} {_flat(resource.retained.target)}"
            if resource.retained
            else "holds nothing"
        )
        rows.append((resource.id, resource.condition + (f" ({resource.fault})" if resource.fault else ""), using, held))
    lines.append(_columns(rows))
    together = "; ".join(" + ".join(group) for group in d.concurrent_groups) or "one resource at a time"
    lines += [
        "",
        "limits",
        _columns(
            [
                ("together", together),
                ("request", f"{limits.max_request_bytes} bytes"),
                (
                    "session",
                    f"{limits.default_session_lifetime_wall_s:g} s by default, "
                    f"{limits.max_session_lifetime_wall_s:g} s at most",
                ),
                (
                    "deadlines",
                    f"{limits.default_max_execution_s:g} s execution and {limits.default_max_wall_s:g} s wall "
                    f"by default, {limits.max_deadline_s:g} s at most",
                ),
                ("events", f"{limits.max_event_subscribers} readers at once"),
                (
                    "video",
                    f"{limits.max_video_viewers} viewers at once; "
                    f"a link must be opened within {limits.video_link_lifetime_s:g} s",
                ),
                ("unavailable", f"persistent stream channel: {d.persistent_channel.reason}"),
            ]
        ),
        "  (homunculus --json describe lists every limit)",
    ]
    return "\n".join(lines)


def _observation_text(observation: Observation, saved: list[Path]) -> str:
    lines = [f"observation {observation.observation_id}  (sequence {observation.sequence}, {observation.context_id})"]
    for rid, sample in observation.samples.items():
        captured = int(sample.capture_time_ns) / 1e9
        validity = "valid" if sample.valid else "INVALID: do not treat as current"
        lines.append(
            f"  {rid}: {validity}, {sample.provenance}, captured at {captured:.3f} s ({sample.clock_id} clock)"
        )
        ordered = sorted(sample.values.items(), key=lambda item: item[0] not in ("positions", "velocities"))
        if ordered:
            lines.append(_columns([(key, _flat(value)) for key, value in ordered], indent=4))
    for image, path in zip(observation.images, saved, strict=True):
        pose = image.camera_pose
        taken = (
            f"lens {_flat(image.focal_px)} px focal, centre {_flat(image.principal_px)} px; camera at "
            f"{_flat(pose.position_m)} m, orientation xyzw {_flat(pose.orientation_xyzw)} in {pose.frame_id} "
            "(optical axes: x right, y down, z forward)"
        )
        lines += [
            f"  image {image.sensor} {image.kind} {image.width}x{image.height} {image.media_type}: {path}",
            textwrap.fill(taken, _width(), initial_indent="    ", subsequent_indent="    "),
        ]
    return "\n".join(lines)


def _session_text(session: Session) -> str:
    if session.state != "active":
        return f"session {session.session_id}: {session.state} ({session.reason})"
    return (
        f"session {session.session_id}: active for {session.principal} in {session.context_id}, "
        f"expires in {session.expires_in_s:.0f} s"
    )


def _goal_text(goal: Goal) -> str:
    action = goal.contract.action
    reason = f" ({goal.result.reason})" if goal.result and goal.result.reason != goal.state else ""
    rows = [("profile", f"{action.profile} on {action.target!r}; resources {', '.join(goal.contract.resources)}")]
    rows.append(("deadlines", f"{goal.contract.max_execution_s:g} s execution, {goal.contract.max_wall_s:g} s wall"))
    if goal.result:
        if goal.result.detail:
            rows.append(("detail", goal.result.detail))
        if goal.result.measurements:
            rows.append(("measured", _flat(goal.result.measurements)))
        if goal.result.completion_observation_id:
            rows.append(("evidence", f"observation {goal.result.completion_observation_id}"))
        if goal.result.stop_confirmed is not None:
            confirmed = "confirmed by measurement" if goal.result.stop_confirmed else "NOT CONFIRMED"
            rows.append(("stop", confirmed))
        if goal.retained:
            held = f"{goal.retained.kind} at {_flat(goal.retained.target)}"
        else:
            held = "unknown: the stop was not confirmed" if goal.result.stop_confirmed is False else "not recorded"
        rows.append(("left holding", held))
    return f"goal {goal.goal_id}: {goal.state}{reason}\n" + _columns(rows)


def _stream_text(stream: Stream) -> str:
    state = stream.state + (
        f" ({stream.closure}{': ' + stream.detail if stream.detail else ''})" if stream.closure else ""
    )
    applied = "none yet" if stream.applied_sequence is None else str(stream.applied_sequence)
    rows = [
        ("profile", f"{stream.profile} on {stream.target!r}"),
        ("sequence", f"accepted {stream.accepted_sequence}, applied {applied}"),
    ]
    if stream.state == "open":
        rows.append(("expires", f"in {stream.expires_in_s:.2f} s unless a newer target arrives"))
    return f"stream {stream.stream_id}: {state}\n" + _columns(rows)


def _operation_text(operation: Operation) -> str:
    reason = f" ({operation.reason}{': ' + operation.detail if operation.detail else ''})" if operation.reason else ""
    rows = [("resources", _flat(operation.resources) if operation.resources else "pending")]
    if operation.outcome:
        rows.append(("outcome", _flat(operation.outcome)))
    if operation.new_context_id:
        rows.append(("new context", f"{operation.new_context_id}; open a new session to continue"))
    return f"{operation.kind} {operation.operation_id}: {operation.state}{reason}\n" + _columns(rows)


# --------------------------------------------------------------------------
# Robot commands
# --------------------------------------------------------------------------


@contextmanager
def _robot(args: argparse.Namespace) -> Iterator[RobotClient]:
    """A client for the selected target. What it learns (session, last goal) is saved as soon as it learns it,
    so another command can act on a goal while this one still waits for it."""
    store = TargetStore()
    with RobotClient(store.get(args.target), on_change=store.save) as client:
        yield client


def _source(path: str) -> str:
    """What messages call an input file: its path, or standard input for `-`."""
    return "standard input" if path == "-" else path


def _read_json(path: str) -> Any:
    source = _source(path)
    try:
        # Bytes, which json.loads decodes itself: UTF-8, with or without a byte-order mark. Standard input is
        # read as file descriptor 0, because sys.stdin is None when the command starts with it closed.
        with open(0 if path == "-" else path, "rb", closefd=path != "-") as file:
            return json.loads(file.read())
    except OSError as error:
        raise UsageError(f"cannot read {source}: {error.strerror}") from None
    except UnicodeDecodeError:
        raise UsageError(f"{source} is not valid JSON: it is not UTF-8 text") from None
    except json.JSONDecodeError as error:
        raise UsageError(
            f"{source} is not valid JSON: {error.msg} at line {error.lineno}, column {error.colno}"
        ) from None


def _request_file(path: str, model: type[BaseModel], payload: str | None = None) -> dict[str, Any]:
    """The fields of a request file: a request as the service takes it, or only its `payload` part.

    A request may leave out what the CLI fills in: its ID, the recorded session and context, and for a
    stream the basis observation and the sequence number. What it does state is sent as written. A field
    that such a request does not have is refused here, since the service would never see it.
    """
    document, source = _read_json(path), _source(path)
    if not isinstance(document, dict):
        raise UsageError(f"{source} must hold a JSON object")
    request = {payload: document} if payload is not None and payload not in document else document
    unknown = sorted(request.keys() - model.model_fields.keys())
    if unknown:
        raise UsageError(f"{source}: unknown field {unknown[0]!r}; a request holds: {', '.join(model.model_fields)}")
    return request


def _parse(model: Any, document: Any, source: str) -> Any:
    try:
        return TypeAdapter(model).validate_python(document)
    except ValidationError as error:
        first = error.errors()[0]
        if first["type"] == "union_tag_invalid":
            known = first["ctx"]["expected_tags"]
            raise UsageError(f"{source}: unknown motion profile {first['ctx']['tag']!r}; known: {known}") from None
        problems = "; ".join(
            f"{'.'.join(str(part) for part in e['loc']) or 'document'}: {e['msg']}" for e in error.errors()[:4]
        )
        raise UsageError(f"{source} is not a valid request: {problems}") from None


def _binding(client: RobotClient, profile: str, on: str | None, mode: str) -> ProfileBinding:
    """Find where this robot offers a profile, from its own description, before sending anything."""
    description = client.describe()
    offered = [b for b in description.profiles if b.profile == profile and mode in b.modes]
    if on is not None:
        offered = [b for b in offered if b.target == on]
    if len(offered) == 1:
        return offered[0]
    everything = ", ".join(f"{b.profile} on {b.target!r} ({'/'.join(b.modes)})" for b in description.profiles)
    if not offered:
        where = f" on {on!r}" if on else ""
        raise RobotError(
            "unsupported_profile",
            f"{description.robot_id} does not offer {profile} as a {mode}{where}; it offers: {everything}",
        )
    choices = ", ".join(b.target for b in offered)
    raise UsageError(f"{profile} is offered on several targets ({choices}); choose one with --on")


def describe(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        description = client.describe()
    _emit(args, description, _description_text(description))
    return OK


def observe(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        observation = client.observe(args.camera, args.after_sequence, args.wait)
        directory = Path(os.path.abspath(Path(args.save).expanduser())) if args.save else TargetStore().home / "images"
        saved = client.save_images(observation, directory)
    document = observation.model_dump(mode="json")
    for image, path in zip(document["images"], saved, strict=True):
        image["file"] = str(path)
    _emit(args, document, _observation_text(observation, saved))
    return OK


def video(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        link = client.video_link(args.camera)
    text = (
        f"camera {link.camera}, live at {link.fps:g} frames a second:\n  {link.url}\n"
        f"Open it in a browser within {link.expires_in_s:g} s. Anyone who has the link can watch, without a token."
    )
    _emit(args, link, text)
    return OK


def session_open(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        session = client.open_session(args.lifetime, args.takeover, args.request_id)
    _emit(args, session, _session_text(session))
    return OK


def session_renew(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        session = client.renew_session(args.lifetime)
    _emit(args, session, _session_text(session))
    return OK


def session_close(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        session = client.close_session()
    _emit(args, session, _session_text(session))
    return OK


def _submit(args: argparse.Namespace, client: RobotClient, action: Action, **settings: Any) -> int:
    """Shared tail of every goal command: check only, or submit and optionally wait for the outcome."""
    settings.setdefault("max_execution_s", args.max_execution)
    settings.setdefault("max_wall_s", args.max_wall)
    if args.dry_run:
        # The goal as it would be submitted, deadlines and start conditions included. A session is not needed.
        proposed = {name: settings[name] for name in GoalSpec.model_fields if name in settings}
        validation = client.validate(goal=GoalSpec(action=action, **proposed))
        text = f"valid: {action.profile} on {action.target!r} would use {', '.join(validation.resources)}"
        _emit(args, validation, text + "\n  nothing was reserved or sent to the robot")
        return OK
    goal = client.submit_goal(action, request_id=args.request_id, **settings)
    if args.wait:
        with _waiting(f"goal {goal.goal_id}", GOAL_GOES_ON):
            goal = client.wait_goal(goal.goal_id, args.timeout)
    hint = "" if goal.state in TERMINAL_GOAL_STATES else "\n  follow it with: homunculus goal inspect --wait"
    _emit(args, goal, _goal_text(goal) + hint)
    return OK if goal.state in ("accepted", "executing", "finishing", "succeeded") else FAILED


def goal_submit(args: argparse.Namespace) -> int:
    request, source = _request_file(args.file, GoalSubmit, "action"), _source(args.file)  # or a bare action
    for field, flag, given in (
        ("request_id", "--request-id", args.request_id),
        ("max_execution_s", "--max-execution", args.max_execution),
        ("max_wall_s", "--max-wall", args.max_wall),
    ):
        if field in request and given is not None:
            raise UsageError(f"{source} holds {field} and {flag} was given as well; give it once")
    action = _parse(Action, request.pop("action"), f"{source}: action")
    if request.get("start_conditions") is not None:
        request["start_conditions"] = _parse(
            StartConditions, request["start_conditions"], f"{source}: start_conditions"
        )
    args.request_id = request.pop("request_id", args.request_id)
    with _robot(args) as client:
        return _submit(args, client, action, **request)


def goal_joints(args: argparse.Namespace) -> int:
    names, positions = [], []
    for assignment in args.joint:
        name, _, value = assignment.partition("=")
        try:
            position = float(value)
        except ValueError:
            position = math.nan
        if not name or not math.isfinite(position):
            raise UsageError(f"expected JOINT=POSITION, got {assignment!r} (for example: joint4=-2.2)")
        names.append(name)
        positions.append(position)
    with _robot(args) as client:
        binding = _binding(client, "joints.trajectory/1", args.on, "goal")
        point = TrajectoryPoint(time_s=args.duration, positions=positions, velocities=[0.0] * len(positions))
        action = JointTrajectoryGoal(profile="joints.trajectory/1", target=binding.target, joints=names, points=[point])
        return _submit(args, client, action)


def goal_gripper(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        binding = _binding(client, "gripper.width/1", args.on, "goal")
        action = GripperWidthGoal(profile="gripper.width/1", target=binding.target, width_m=args.width)
        return _submit(args, client, action)


def goal_hand(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        binding = _binding(client, "hand.posture/1", args.on, "goal")
        action = HandPostureGoal(profile="hand.posture/1", target=binding.target, posture=args.posture)
        return _submit(args, client, action)


def goal_pose(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        binding = _binding(client, "tool.pose.linear/1", args.on, "goal")
        action = ToolPoseGoal(
            profile="tool.pose.linear/1",
            target=binding.target,
            frame_id=args.frame,
            position_m=args.position,
            orientation_xyzw=args.orientation,
            speed_m_s=args.speed,
            position_tolerance_m=args.position_tolerance,
            orientation_tolerance_rad=args.orientation_tolerance,
        )
        return _submit(args, client, action)


def goal_base(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        binding = _binding(client, "base.planar_velocity/1", args.on, "goal")
        action = BaseVelocityGoal(
            profile="base.planar_velocity/1",
            target=binding.target,
            forward_m_s=args.forward or 0.0,
            left_m_s=args.left or 0.0,
            yaw_rate_rad_s=args.yaw or 0.0,
            duration_s=args.duration,
        )
        return _submit(args, client, action)


def goal_inspect(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        goal = client.goal(args.goal_id)
        if args.wait:
            with _waiting(f"goal {goal.goal_id}", GOAL_GOES_ON):
                goal = client.wait_goal(goal.goal_id, args.timeout)
    _emit(args, goal, _goal_text(goal))
    return OK if goal.state in ("accepted", "executing", "finishing", "succeeded") else FAILED


def goal_cancel(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        goal = client.cancel_goal(args.goal_id)
        if goal.state == "stopping" and not args.no_wait:
            with _waiting(f"goal {goal.goal_id}", "the goal is being stopped; see it with `homunculus goal inspect`"):
                goal = client.wait_goal(goal.goal_id, args.timeout)
    _emit(args, goal, _goal_text(goal))
    return FAILED if goal.result and goal.result.stop_confirmed is False else OK


def _only_file(args: argparse.Namespace, *flags: str) -> None:
    """A file states the whole request. A flag given beside it would be ignored, so it is refused instead."""
    given = [flag for flag in flags if getattr(args, flag.lstrip("-")) is not None]
    if given:
        raise UsageError(f"--file holds the whole request; give it without {', '.join(given)}")


def _velocity(args: argparse.Namespace) -> BaseVelocityTarget:
    return BaseVelocityTarget(
        forward_m_s=args.forward or 0.0, left_m_s=args.left or 0.0, yaw_rate_rad_s=args.yaw or 0.0
    )


def stream_open(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        # What the request names beside the stream itself: its ID, session, context, and basis observation.
        named: dict[str, Any] = {"basis_observation_id": args.basis}
        if args.file:
            _only_file(args, "--forward", "--left", "--yaw", "--validity", "--on", "--basis")
            request = _request_file(args.file, StreamOpen)
            named = {name: request.pop(name) for name in list(request) if name not in StreamSpec.model_fields}
            spec = _parse(StreamSpec, request, _source(args.file))
        else:
            binding = _binding(client, "base.planar_velocity/1", args.on, "stream")
            spec = StreamSpec(
                profile=binding.profile,
                target=binding.target,
                update_validity_wall_s=1.0 if args.validity is None else args.validity,
                initial_target=_velocity(args),
            )
        if args.dry_run:
            validation = client.validate(stream=spec)
            _emit(args, validation, f"valid: a {spec.profile} stream on {spec.target!r}; nothing was opened")
            return OK
        stream = client.open_stream(
            spec.profile, spec.target, spec.initial_target, spec.update_validity_wall_s, **named
        )
    hint = f"\n  send the next target within {stream.expires_in_s:.2f} s: homunculus stream update --forward ..."
    _emit(args, stream, _stream_text(stream) + hint)
    return OK


def stream_update(args: argparse.Namespace) -> int:
    # What the update names beside its target: the session, context, sequence, and basis observation.
    named: dict[str, Any] = {"basis_observation_id": args.basis}
    setpoint = _velocity(args)
    if args.file:
        _only_file(args, "--forward", "--left", "--yaw", "--basis")
        named = _request_file(args.file, StreamUpdate, "target")  # or a bare target
        setpoint = _parse(BaseVelocityTarget, named.pop("target"), f"{_source(args.file)}: target")
    with _robot(args) as client:
        result = client.update_stream(setpoint, stream_id=args.stream_id, **named)
    note = "duplicate of an earlier update; expiry unchanged" if result.duplicate else "accepted"
    _emit(args, result, f"sequence {result.ack.sequence} {note}\n" + _stream_text(result.stream))
    return OK


def stream_inspect(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        stream = client.stream(args.stream_id)
    _emit(args, stream, _stream_text(stream))
    return OK


def stream_close(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        stream = client.close_stream(args.stream_id)
        with _waiting(f"stream {stream.stream_id}", "its stop goes on; see it with `homunculus stream inspect`"):
            deadline = time.monotonic() + args.timeout
            while stream.state == "stopping" and time.monotonic() < deadline:
                time.sleep(0.05)
                stream = client.stream(stream.stream_id)
    _emit(args, stream, _stream_text(stream))
    # A stream that had ended by itself is closed, as asked. What fails is a stop that was not confirmed.
    return OK if stream.state == "closed" and stream.closure != "failed" else FAILED


def _finish_operation(args: argparse.Namespace, client: RobotClient, operation: Operation) -> int:
    if not args.no_wait:
        advice = "the operation goes on; see it with `homunculus operation inspect`"
        with _waiting(f"operation {operation.operation_id}", advice):
            operation = client.wait_operation(operation.operation_id, args.timeout)
    _emit(args, operation, _operation_text(operation))
    return FAILED if operation.state == "failed" else OK


def stop(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        return _finish_operation(args, client, client.stop(as_operator=args.operator))


def manage(args: argparse.Namespace) -> int:
    parameters = _read_json(args.file) if args.file else {}
    if not isinstance(parameters, dict):
        raise UsageError(f"{_source(args.file)} must hold a JSON object of parameters")
    for assignment in args.param:
        key, separator, value = assignment.partition("=")
        if not separator:
            raise UsageError(f"expected KEY=VALUE, got {assignment!r}")
        try:
            parameters[key] = json.loads(value)
        except json.JSONDecodeError:
            parameters[key] = value
    with _robot(args) as client:
        return _finish_operation(args, client, client.manage(args.operation, parameters, args.request_id))


def operation_inspect(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        operation = client.operation(args.operation_id)
    _emit(args, operation, _operation_text(operation))
    return FAILED if operation.state == "failed" else OK


def events(args: argparse.Namespace) -> int:
    with _robot(args) as client:
        for event in client.events(args.after, args.follow):
            if args.json:
                print(event.model_dump_json(), flush=True)  # one JSON document per line
            else:
                print(
                    f"{event.sequence:>5}  {int(event.time_ns) / 1e9:12.3f}  {event.type:<22} {_flat(event.data)}",
                    flush=True,
                )
    return OK


# --------------------------------------------------------------------------
# Targets and local services
# --------------------------------------------------------------------------


def connect(args: argparse.Namespace) -> int:
    store = TargetStore()
    if args.home and (args.url or args.token or args.operator_token):
        raise UsageError(
            "--home reads the URL and the tokens from that directory; give it alone, or a URL with --token"
        )
    if args.home:
        added = store.connect_home(Path(args.home).expanduser().resolve())
    elif args.url and args.token:
        added = store.connect(args.url.rstrip("/"), args.token, args.operator_token)
    else:
        raise UsageError("give a service URL with --token, or --home DIR for a service on this machine")
    store.save()
    text = "\n".join(f"added target {t.name} ({t.robot_id} at {t.url})" for t in added)
    _emit(args, [t.name for t in added], text + f"\ndefault target: {store.default}")
    return OK


def list_rigs(args: argparse.Namespace) -> int:
    shipped = {name: rigs.load(rigs.SHIPPED / name) for name in rigs.names()}
    rows = [(name, rig.menagerie or "in the rig's folder", rig.description) for name, rig in shipped.items()]
    text = (
        _columns([("RIG", "ROBOT", ""), *rows], indent=0)
        + "\n\nA robot named in the ROBOT column is downloaded from MuJoCo Menagerie on first use.\n"
        "check one and download its robot:  homunculus install RIG\n"
        "start one:                         homunculus demo start RIG\n"
        "a rig folder of your own is named by its path, such as ./my-arm"
    )
    _emit(args, {"rigs": [{"name": name, **rig.model_dump()} for name, rig in shipped.items()]}, text)
    return OK


def install(args: argparse.Namespace) -> int:
    folder = rigs.find(args.rig)
    rig = rigs.load(folder)
    try:
        from .backends.mujoco import check  # MuJoCo loads only for the commands that need it
    except ImportError as error:
        raise StartupError(f"cannot check a rig: {error}\n{rigs.NEEDS_MUJOCO}") from None
    images = Path(os.path.abspath(Path(args.save).expanduser())) if args.save else TargetStore().home / "images"
    steps = check(folder, images)
    passed = all(step.passed for step in steps)
    text = (
        f"{folder.name}: {rig.description}\n"
        + _columns([("ok" if step.passed else "FAILED", step.name, step.detail) for step in steps])
        + (
            f"\nready: start it with `homunculus demo start {args.rig}`"
            if passed
            else "\nnot ready: a service that serves this rig has the fault that the failed step shows"
        )
    )
    document = {
        "rig": folder.name,
        "folder": str(folder),
        "passed": passed,
        "steps": [dataclasses.asdict(step) for step in steps],
    }
    _emit(args, document, text)
    return OK if passed else FAILED


def targets(args: argparse.Namespace) -> int:
    store = TargetStore()
    document = {
        "default": store.default,
        "targets": [
            {"name": t.name, "robot_id": t.robot_id, "url": t.url, "session_id": t.session_id}
            for t in store.targets.values()
        ],
    }
    rows = [
        ("*" if t.name == store.default else "", t.name, t.robot_id, t.url, "session open" if t.session_id else "")
        for t in store.targets.values()
    ]
    header = [("", "TARGET", "ROBOT", "SERVICE", "")]
    empty = "no targets; start the local demo with `homunculus demo start`"
    text = _columns(header + rows, indent=0) if rows else empty
    _emit(args, document, text)
    return OK


def use(args: argparse.Namespace) -> int:
    store = TargetStore()
    store.default = store.get(args.name).name
    store.save()
    _emit(args, {"default": store.default}, f"default target: {store.default}")
    return OK


def forget(args: argparse.Namespace) -> int:
    store = TargetStore()
    removed = store.forget([store.get(args.name).name])
    store.save()
    text = f"forgot target {removed[0]}; nothing was sent to its service"
    _emit(args, {"removed": removed, "default": store.default}, text + _no_default(store))
    return OK


def _no_default(store: TargetStore) -> str:
    """What a command adds when it removed the default target and others remain: none is chosen for the user."""
    if store.default is None and store.targets:
        return "\nno default target is selected now; choose one with `homunculus use NAME`"
    return ""


def _backend(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    """The backend that a `serve` or `demo start` command line names, and its options.

    RIG is short for the mujoco backend serving that rig. Without RIG or --backend, the backend is fake.
    """
    options = dict(args.backend_arg)
    if len(options) < len(args.backend_arg):
        raise UsageError("give each --backend-arg KEY once")
    if args.rig is None:
        return args.backend or "fake", options
    if args.backend not in (None, "mujoco") or "rig" in options:
        raise UsageError(
            f"{args.rig!r} names the rig for the mujoco backend to serve; "
            "give it without another --backend and without --backend-arg rig="
        )
    return "mujoco", {**options, "rig": args.rig}


def _serve_options(args: argparse.Namespace) -> list[str]:
    """The options `homunculus demo start` gives the `homunculus serve` it starts.

    That service runs in this directory, so the path of a rig folder means the same to it.
    """
    backend, backend_options = _backend(args)
    options = ["--backend", backend, "--port", str(args.port)]
    for key, value in backend_options.items():
        options += ["--backend-arg", f"{key}={value}"]
    return options


def serve(args: argparse.Namespace) -> int:
    from .server import HomeInUse  # the service stack loads only for this command
    from .server import serve as run_service

    backend, options = _backend(args)
    try:
        run_service(
            backend,
            options,
            Path(os.path.abspath(Path(args.home).expanduser())) if args.home else TargetStore().home / "service",
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            demo_attempt=args.demo_attempt,
        )
    except HomeInUse as error:  # exit status NO_SERVICE, which `homunculus demo start` relies on
        raise ClientError(str(error), hint="stop it first, or give this service its own --home") from None
    except OSError as error:
        # Status 1, like any other failure to start. NO_SERVICE stays reserved for a home that is in use:
        # `homunculus demo start` reads that status as "these files belong to another service".
        raise StartupError(_file_problem(error)) from None
    return OK


def _file_problem(error: OSError) -> str:
    return f"cannot use {error.filename}: {error.strerror}" if error.filename else str(error)


def _service_record(home: Path) -> dict[str, Any]:
    """What the service running in `home` wrote about itself; empty when there is none."""
    try:
        return json.loads((home / "service.json").read_text())
    except OSError, ValueError:  # absent, or read while it is being written
        return {}


def _demo_service(home: Path) -> dict[str, Any] | None:
    """The demo's service record, only if that same service still answers."""
    record = _service_record(home)
    try:
        health = httpx.get(f"{record['url']}/healthz", timeout=2.0, trust_env=False).json()  # no proxy
    except KeyError, ValueError, httpx.TransportError:
        return None
    return record if health.get("boot_id") == record.get("boot_id") else None


def _demo(home: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The demo's service record if it answers; otherwise advice if a live process still holds its files.

    Only the journal's lock can tell that such a process exists. Which process it is cannot be proven from
    here, so the demo commands neither signal it nor remove files from under it.
    """
    record = _demo_service(home)
    if record or not journal.held(home / "journal.sqlite"):
        return record, None
    pid = _service_record(home).get("pid")
    if pid is None:
        return None, "it has no service record, so it is probably starting or stopping: try again in a moment"
    advice = (
        f"its record names process {pid}: if that is the stuck service (`ps -p {pid}`), end it; "
        "`homunculus demo stop` then removes its files"
    )
    return None, advice


def _end(process: subprocess.Popen[bytes]) -> None:
    """End the service this command started, which is its child: told to stop, it removes its own record."""
    process.terminate()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _terminate(pid: int, database: Path, patience_s: float = 15.0) -> None:
    """Ask a service another command started to shut down, and wait until it is gone; kill it only if it stays."""
    deadline = time.monotonic() + patience_s
    with suppress(ProcessLookupError):  # raised by the first signal sent after the process is gone
        os.kill(pid, signal.SIGTERM)
        while time.monotonic() < deadline:
            if not journal.held(database):
                return
            time.sleep(0.05)
        if journal.held(database):
            os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while journal.held(database):
        if time.monotonic() >= deadline:
            raise ClientError(f"process {pid} still holds the demo's journal")
        time.sleep(0.05)


def demo_start(args: argparse.Namespace) -> int:
    store = TargetStore()
    home = store.home / "demo"
    boot_id = "boot-" + secrets.token_hex(8)
    claim_file = home / "start.json"
    log_path = home / "server.log"
    command = [
        sys.executable,
        "-m",
        "homunculus.cli",
        "serve",
        "--home",
        str(home),
        "--demo-attempt",
        boot_id,
        *_serve_options(args),
    ]
    stopped = ClientError(
        "the demo was stopped while it was starting", hint="start it again with `homunculus demo start`"
    )
    deadline = time.monotonic() + args.startup_timeout
    process = None
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        with journal.lock_directory(home.parent):
            record, advice = _demo(home)
            if record or advice:
                raise ClientError("the demo service is already starting or running", hint=advice)
            starting = journal.DemoStart.read(home)
            if starting is not None:
                try:
                    os.kill(starting.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ClientError("the demo service is already starting")
            home.mkdir(parents=True, exist_ok=True)
            with open(log_path, "ab") as log_file:
                logged = log_file.tell()
                process = subprocess.Popen(
                    command, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
                )
            claim = journal.DemoStart(boot_id, process.pid)
            claim_file.write_text(json.dumps(dataclasses.asdict(claim)))

        while (record := _service_record(home)).get("boot_id") != boot_id:
            if journal.DemoStart.read(home) != claim:
                raise stopped
            if process.poll() is not None or time.monotonic() > deadline:
                problem = "exited during startup" if process.poll() is not None else "did not start in time"
                output = log_path.read_bytes()[logged:].decode(errors="replace")[-2000:].strip()
                hint = None if output else "allow more time with --startup-timeout"
                raise ClientError(f"the demo service {problem}" + (f":\n{output}" if output else ""), hint=hint)
            time.sleep(0.05)
        with journal.lock_directory(home.parent):
            if journal.DemoStart.read(home) != claim or process.poll() is not None:
                raise stopped
            store = TargetStore(home.parent)
            store.forget_home(home)
            added = store.connect_home(home)
            store.default = added[-1].name
            store._save_locked()
            claim_file.unlink()
    except BaseException:
        if process is not None:
            _end(process)
            with journal.lock_directory(home.parent):
                if journal.DemoStart.read(home) == journal.DemoStart(boot_id, process.pid):
                    if journal.held(home / "journal.sqlite"):
                        claim_file.unlink()
                    else:
                        shutil.rmtree(home)
                        store = TargetStore(home.parent)
                        store.forget_home(home)
                        store._save_locked()
        raise
    names = ", ".join(t.name for t in added)
    text = (
        f"demo service started ({record['backend']} backend) at {record['url']}, pid {record['pid']}\n"
        f"targets: {names}   (default: {store.default}; choose another with --target or `homunculus use`)\n"
        f"files: {home}\n\n"
        "try:\n"
        "  homunculus describe\n"
        "  homunculus session open\n"
        "  homunculus observe\n"
        "stop it and remove its files with:\n"
        "  homunculus demo stop"
    )
    _emit(args, {**record, "home": str(home), "targets": [t.name for t in added], "default": store.default}, text)
    return OK


def demo_stop(args: argparse.Namespace) -> int:
    store = TargetStore()
    home = store.home / "demo"
    with journal.lock_directory(home.parent):
        record, advice = _demo(home)
        if record:
            _terminate(record["pid"], home / "journal.sqlite")
        elif advice:
            raise ClientError(
                f"a service holds the demo's files ({home}) but does not answer; nothing was stopped or removed",
                hint=advice,
            )
        try:
            shutil.rmtree(home)
            existed = True
        except FileNotFoundError:
            existed = False
        store = TargetStore(home.parent)
        removed = store.forget_home(home)
        store._save_locked()
    stopped = f"stopped the demo service (pid {record['pid']})" if record else "no demo service was running"
    cleaned = (
        f"removed {home} and targets {', '.join(removed) or 'none'}" if existed or removed else "nothing to remove"
    )
    result = {"stopped": bool(record), "removed_targets": removed, "removed_home": existed, "default": store.default}
    _emit(args, result, f"{stopped}\n{cleaned}" + _no_default(store))
    return OK


def demo_status(args: argparse.Namespace) -> int:
    store = TargetStore()
    record, advice = _demo(store.home / "demo")
    if record is None:
        text = (
            f"a service holds the demo's files but does not answer\n  {advice}"
            if advice
            else "the demo service is not running; start it with `homunculus demo start`"
        )
        _emit(args, {"running": False, "home_in_use": advice is not None}, text)
        return FAILED
    home = store.home / "demo"
    names = [t.name for t in store.targets.values() if t.home == str(home)]
    # A start that was killed outright leaves a service without targets.
    saved = ", ".join(names) or (
        f"none saved; add them with `homunculus connect --home {home}`, or `homunculus demo stop`"
    )
    service = f"demo service running ({record['backend']} backend) at {record['url']}, pid {record['pid']}"
    _emit(args, {"running": True, **record, "targets": names}, f"{service}\ntargets: {saved}")
    return OK


def version(args: argparse.Namespace) -> int:
    _emit(args, {"version": __version__, "protocol": PROTOCOL}, f"homunculus {__version__} (protocol {PROTOCOL})")
    return OK


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------


def _port(text: str) -> int:
    if not text.isdecimal() or int(text) > 65535:
        raise argparse.ArgumentTypeError(f"{text!r} is not a port (0 to 65535; 0 takes any free port)")
    return int(text)


def _host(text: str) -> str:
    """Where `homunculus serve` listens. An empty name would mean every interface, and the service has no IPv6."""
    if not text or text != text.strip() or ":" in text:
        raise argparse.ArgumentTypeError(f"{text!r} is not an IPv4 address or a host name")
    return text


def _identifier(text: str) -> str:
    """An ID or target name as typed. An empty one, as from a shell variable that is not set, names nothing: it is
    refused here, before anything is sent, instead of reaching the service or standing for the last one."""
    if not text.strip():
        raise argparse.ArgumentTypeError("it is empty; if a shell variable should hold it, that variable is not set")
    return text


def _backend_arg(text: str) -> tuple[str, str]:
    key, equals, value = text.partition("=")
    if not key or not equals:
        raise argparse.ArgumentTypeError(f"{text!r} is not KEY=VALUE")
    return key, value


def _seconds(text: str) -> float:
    """A wait that this command applies itself: zero or more seconds, or inf for no limit."""
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not value >= 0:  # nan too: it compares false with every clock reading, so the wait would never end
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds (0 or more)")
    return value


class _Parser(argparse.ArgumentParser):
    """Reports an unusable command line the way every other error is reported: as JSON under --json."""

    json_errors = False  # set by main() from the raw arguments, since parsing is the thing that failed

    def error(self, message: str) -> NoReturn:
        if self.json_errors:
            hint = f"see `{self.prog} --help`"
            self.exit(
                USAGE, json.dumps({"error": {"code": "usage", "message": message, "hint": hint, "details": {}}}) + "\n"
            )
        super().error(message)


def _parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "-t",
        "--target",
        type=_identifier,
        default=argparse.SUPPRESS,
        help="target to use (default: the one `homunculus use` chose, or $HOMUNCULUS_TARGET)",
    )
    shared.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="print the result as JSON")

    parser = _Parser(  # subcommand parsers are created from the same class
        prog="homunculus",
        usage="homunculus [-t TARGET] [--json] COMMAND ...",
        description="Discover a robot's capabilities, read observations, and request motion through one interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-t", "--target", type=_identifier, help="target to use, for example fake:arm")
    parser.add_argument("--json", action="store_true", default=False, help="print the result as JSON")
    parser.add_argument("--version", action="version", version=f"homunculus {__version__} (protocol {PROTOCOL})")
    # prog: subcommands are named `homunculus goal cancel`, not after the usage line above.
    commands = parser.add_subparsers(
        dest="command", metavar="COMMAND", required=True, help=argparse.SUPPRESS, prog="homunculus"
    )
    summaries: dict[str, str] = {}  # top-level commands, listed by section in the epilog below

    def command(group: Any, name: str, run: Any, summary: str) -> argparse.ArgumentParser:
        if group is commands:
            summaries[name] = summary
            leaf = group.add_parser(name, parents=[shared], description=summary)
        else:
            leaf = group.add_parser(name, parents=[shared], help=summary, description=summary)
        leaf.set_defaults(run=run)
        return leaf

    def group(name: str, summary: str) -> Any:
        summaries[name] = summary
        node = commands.add_parser(name, description=summary)
        return node.add_subparsers(dest=f"{name}_command", metavar="ACTION", required=True)

    def backend_options(leaf: argparse.ArgumentParser) -> None:
        leaf.add_argument(
            "rig",
            nargs="?",
            metavar="RIG",
            help="a robot body for the mujoco backend to serve: a name from `homunculus rigs`, such as so101, "
            "or the path of a rig folder, such as ./my-arm",
        )
        leaf.add_argument(
            "--backend",
            metavar="NAME",
            help="fake, mujoco, task, or MODULE:FUNCTION for a backend of your own "
            "(default: mujoco with a RIG, otherwise fake)",
        )
        leaf.add_argument(
            "--backend-arg",
            type=_backend_arg,
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="an option for the backend, such as robot_id=left for mujoco; repeat for more",
        )
        leaf.add_argument(
            "--port", type=_port, default=0, metavar="N", help="port to listen on (default: any free port)"
        )

    def waiting(leaf: argparse.ArgumentParser, default: float = 60.0) -> None:
        leaf.add_argument(
            "--timeout",
            type=_seconds,
            default=default,
            metavar="S",
            help=f"longest wait in seconds (default: {default:g})",
        )

    demo = group("demo", "Run a local demo service: start, stop, status.")
    leaf = command(demo, "start", demo_start, "Start a local service in the background and add its robots as targets.")
    backend_options(leaf)
    leaf.add_argument(
        "--startup-timeout",
        type=_seconds,
        default=180.0,
        metavar="S",
        help="longest wait for the service to start, in seconds (default: 180)",
    )
    command(demo, "stop", demo_stop, "Stop the demo service and remove its files and targets.")
    command(demo, "status", demo_status, "Say whether the demo service is running.")

    leaf = command(commands, "serve", serve, "Run a service in the foreground until interrupted.")
    leaf.add_argument("--demo-attempt", help=argparse.SUPPRESS)
    backend_options(leaf)
    leaf.add_argument(
        "--home",
        metavar="DIR",
        help="directory for the journal, tokens, and service record (default: under $HOMUNCULUS_HOME)",
    )
    leaf.add_argument(
        "--host",
        type=_host,
        default="127.0.0.1",
        metavar="ADDRESS",
        help="IPv4 address or host name to listen on (default: 127.0.0.1)",
    )
    leaf.add_argument(
        "--log-level",
        default="warning",
        choices=("debug", "info", "warning", "error"),
        help="how much the service logs to standard error (default: warning)",
    )

    command(commands, "rigs", list_rigs, "List the robot bodies that ship with homunculus.")
    leaf = command(commands, "install", install, "Download a rig's robot and check that the rig works here.")
    leaf.epilog = (
        "The check loads the rig, then asks it to hold its start pose, to move each arm joint to a goal and "
        "back, to close and open its gripper, and to take a picture with each of its cameras. It shows that "
        "the rig is wired correctly. It does not show that the arm is strong enough for a task or that a camera "
        "is aimed well: look at the pictures. Once a rig's robot is downloaded, starting it needs no network."
    )
    leaf.add_argument(
        "rig", metavar="RIG", help="a name from `homunculus rigs`, or the path of a rig folder, such as ./my-arm"
    )
    leaf.add_argument(
        "--save",
        metavar="DIR",
        help="save the cameras' pictures into DIR (default: $HOMUNCULUS_HOME/images, ~/.homunculus/images if unset)",
    )

    leaf = command(commands, "connect", connect, "Add the robots of a running service as targets.")
    leaf.add_argument("url", nargs="?", metavar="URL", help="service URL, for example http://127.0.0.1:8000")
    leaf.add_argument("--token", help="client access token")
    leaf.add_argument("--operator-token", help="operator access token, for stop without a session and management")
    leaf.add_argument("--home", metavar="DIR", help="read the URL and tokens from a local service's home directory")
    command(commands, "targets", targets, "List known targets.")
    leaf = command(commands, "use", use, "Choose the default target.")
    leaf.add_argument("name", type=_identifier, metavar="NAME", help="a name from `homunculus targets`")
    leaf = command(commands, "forget", forget, "Remove a saved target.")
    leaf.add_argument("name", type=_identifier, metavar="NAME", help="a name from `homunculus targets`")

    command(commands, "describe", describe, "Show what the robot supports and its current state.")
    leaf = command(commands, "observe", observe, "Read the latest measurements (and camera images).")
    leaf.epilog = (
        "Values are in metres, radians, seconds, newtons, and newton-metres. A rotating joint is measured in "
        "radians and a sliding joint in metres; `homunculus describe` lists the unit of every joint that takes "
        "positions."
    )
    leaf.add_argument(
        "--camera",
        type=_identifier,
        action="append",
        default=[],
        metavar="NAME",
        help="also record this camera, one of the robot's own that `homunculus describe` lists (repeatable)",
    )
    leaf.add_argument(
        "--save",
        metavar="DIR",
        help="save recorded images into DIR (default: $HOMUNCULUS_HOME/images, ~/.homunculus/images if unset)",
    )
    leaf.add_argument("--after-sequence", type=int, metavar="N", help="wait for measurements newer than observation N")
    leaf.add_argument(
        "--wait", type=_seconds, default=2.0, metavar="S", help="longest wait with --after-sequence (default: 2)"
    )
    leaf = command(commands, "video", video, "Print a link that plays a camera live in a browser.")
    leaf.add_argument(
        "camera", type=_identifier, metavar="CAMERA", help="a camera that `homunculus describe` lists with live video"
    )

    session = group("session", "Open, renew, or close the control session.")
    leaf = command(session, "open", session_open, "Acquire the control session. Nothing moves.")
    leaf.add_argument("--lifetime", type=float, metavar="S", help="lifetime in seconds (default: the service's)")
    leaf.add_argument(
        "--takeover",
        action="store_true",
        help="operator only: revoke the current holder first; the new session is the operator's",
    )
    leaf.add_argument(
        "--request-id",
        type=_identifier,
        metavar="ID",
        help="reuse an ID to learn the session of an unanswered attempt",
    )
    leaf = command(session, "renew", session_renew, "Extend the session. Goals and streams are not extended.")
    leaf.add_argument("--lifetime", type=float, metavar="S", help="new lifetime in seconds (default: the service's)")
    command(session, "close", session_close, "Release authority and stop what this session commanded.")

    def velocity_options(leaf: argparse.ArgumentParser) -> None:
        leaf.add_argument("--forward", type=float, metavar="M_S", help="forward speed in m/s (default: 0)")
        leaf.add_argument("--left", type=float, metavar="M_S", help="leftward speed in m/s (default: 0)")
        leaf.add_argument(
            "--yaw", type=float, metavar="RAD_S", help="turn rate in rad/s, positive to the left (default: 0)"
        )

    goal = group("goal", "Finite movements: joints, gripper, hand, base, submit.")
    last_goal = "default: the last goal submitted to this target"

    def goal_options(leaf: argparse.ArgumentParser, on: bool = True) -> None:
        if on:
            leaf.add_argument(
                "--on",
                type=_identifier,
                metavar="NAME",
                help="which of the robot's targets to command, if several offer this",
            )
        leaf.add_argument("--wait", action="store_true", help="wait for the outcome; exit 1 unless it succeeded")
        waiting(leaf)
        leaf.add_argument(
            "--dry-run", action="store_true", help="check the request without reserving or moving anything"
        )
        leaf.add_argument("--max-execution", type=float, metavar="S", help="execution-time deadline")
        leaf.add_argument("--max-wall", type=float, metavar="S", help="wall-time deadline")
        leaf.add_argument(
            "--request-id",
            type=_identifier,
            metavar="ID",
            help="reuse an ID to retry a request without running it twice",
        )

    leaf = command(goal, "submit", goal_submit, "Submit a goal from a JSON file: an action, or a full request.")
    leaf.add_argument("--file", required=True, metavar="PATH", help="JSON file, or - for standard input")
    goal_options(leaf, on=False)
    leaf = command(goal, "joints", goal_joints, "Move named joints to positions (joints.trajectory/1).")
    leaf.add_argument("joint", nargs="+", metavar="JOINT=POSITION", help="for example joint4=-2.2 (radians or metres)")
    leaf.add_argument("--duration", type=float, default=2.0, metavar="S", help="time to arrive (default: 2)")
    goal_options(leaf)
    leaf = command(goal, "gripper", goal_gripper, "Move a parallel gripper to a width (gripper.width/1).")
    leaf.add_argument("width", type=float, metavar="WIDTH_M", help="gap between the fingers in metres")
    goal_options(leaf)
    leaf = command(goal, "hand", goal_hand, "Move a hand to a named posture (hand.posture/1).")
    leaf.add_argument("posture", metavar="POSTURE", help="a posture the robot advertises, for example fist")
    goal_options(leaf)
    leaf = command(
        goal, "pose", goal_pose, "Move a tool, such as a hand, in a straight line to a pose (tool.pose.linear/1)."
    )
    leaf.add_argument("position", nargs=3, type=float, metavar=("X", "Y", "Z"), help="position in metres")
    leaf.add_argument(
        "--orientation",
        nargs=4,
        type=float,
        required=True,
        metavar=("QX", "QY", "QZ", "QW"),
        help="unit quaternion, x y z w order",
    )
    leaf.add_argument("--frame", default="world", help="frame of the pose (default: world)")
    leaf.add_argument("--speed", type=float, metavar="M_S", help="peak speed (default: the robot's)")
    leaf.add_argument("--position-tolerance", type=float, metavar="M", help="how close counts as reached")
    leaf.add_argument("--orientation-tolerance", type=float, metavar="RAD", help="how close counts as reached")
    goal_options(leaf)
    leaf = command(goal, "base", goal_base, "Travel at one velocity for a time, then stop (base.planar_velocity/1).")
    velocity_options(leaf)
    leaf.add_argument(
        "--duration",
        type=float,
        required=True,
        metavar="S",
        help="travel time in seconds; the goal then stops the base",
    )
    goal_options(leaf)
    leaf = command(goal, "inspect", goal_inspect, "Show a goal's state and result (default: the last one submitted).")
    leaf.add_argument("goal_id", nargs="?", type=_identifier, metavar="GOAL_ID", help=last_goal)
    leaf.add_argument("--wait", action="store_true", help="wait until it finishes; exit 1 unless it succeeded")
    waiting(leaf)
    leaf = command(goal, "cancel", goal_cancel, "Stop one goal. Other activity continues.")
    leaf.add_argument("goal_id", nargs="?", type=_identifier, metavar="GOAL_ID", help=last_goal)
    leaf.add_argument("--no-wait", action="store_true", help="return before the stop is confirmed")
    waiting(leaf, 15.0)

    stream = group("stream", "Continuous velocity targets: open, update, close.")

    def setpoint_options(leaf: argparse.ArgumentParser, request: str) -> None:
        velocity_options(leaf)
        leaf.add_argument(
            "--basis",
            type=_identifier,
            metavar="OBSERVATION_ID",
            help=(
                "the observation this target was decided on: the target is valid for the stream's window "
                "from the time that observation was issued (default: an observation read now)"
            ),
        )
        leaf.add_argument(
            "--file", metavar="PATH", help=f"JSON file with {request}, in place of the flags, or - for standard input"
        )

    last_stream = "default: the last stream opened in the recorded session"
    leaf = command(stream, "open", stream_open, "Open a velocity stream with its first target.")
    setpoint_options(leaf, "a stream request")
    leaf.add_argument("--validity", type=float, metavar="S", help="how long each target stays valid (default: 1)")
    leaf.add_argument("--on", type=_identifier, metavar="NAME", help="which of the robot's targets to command")
    leaf.add_argument("--dry-run", action="store_true", help="check the request without opening anything")
    leaf = command(stream, "update", stream_update, "Replace the stream's target. Nothing stops in between.")
    leaf.add_argument("stream_id", nargs="?", type=_identifier, metavar="STREAM_ID", help=last_stream)
    setpoint_options(leaf, "a target or an update request")
    leaf = command(stream, "inspect", stream_inspect, "Show accepted and applied sequence, expiry, and state.")
    leaf.add_argument("stream_id", nargs="?", type=_identifier, metavar="STREAM_ID", help=last_stream)
    leaf = command(stream, "close", stream_close, "Close the stream, which stops the motion it commanded.")
    leaf.add_argument("stream_id", nargs="?", type=_identifier, metavar="STREAM_ID", help=last_stream)
    waiting(leaf, 15.0)

    leaf = command(commands, "stop", stop, "Stop everything this session commands.")
    leaf.add_argument("--operator", action="store_true", help="stop as operator: needs no session or current context")
    leaf.add_argument("--no-wait", action="store_true", help="return before stopping is confirmed")
    waiting(leaf, 30.0)
    leaf = command(commands, "manage", manage, "Run an operator operation, such as recover or world.reset.")
    leaf.add_argument("operation", metavar="OPERATION", help="see `homunculus describe` for this robot's operations")
    leaf.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="one parameter; VALUE is read as JSON when it parses, else as text (repeatable)",
    )
    leaf.add_argument("--file", metavar="PATH", help="JSON object of parameters")
    leaf.add_argument(
        "--request-id",
        type=_identifier,
        metavar="ID",
        help="reuse an ID to retry a request without running it twice",
    )
    leaf.add_argument("--no-wait", action="store_true", help="return before the operation finishes")
    waiting(leaf, 120.0)
    operation = group("operation", "Inspect a stop or management operation.")
    leaf = command(operation, "inspect", operation_inspect, "Show an operation's progress and outcome.")
    leaf.add_argument(
        "operation_id",
        nargs="?",
        type=_identifier,
        metavar="OPERATION_ID",
        help="default: the last stop or management operation on this target",
    )
    leaf = command(commands, "events", events, "Print numbered runtime events (--follow to keep reading).")
    leaf.add_argument("--after", type=int, default=0, metavar="N", help="only events numbered above N")
    leaf.add_argument("--follow", action="store_true", help="keep printing new events until interrupted")
    command(commands, "version", version, "Print the version.")

    sections = {
        "Get started": ["demo", "rigs", "install", "targets", "use", "connect", "forget", "serve"],
        "Read the robot": ["describe", "observe", "video", "events"],
        "Move it": ["session", "goal", "stream"],
        "Stop and recover": ["stop", "manage", "operation"],
        "Other": ["version"],
    }
    listed = itertools.chain.from_iterable(sections.values())
    assert sorted(listed) == sorted(summaries), "every command is listed once"
    width = max(map(len, summaries)) + 2
    listing = [
        f"{title}:\n" + "\n".join(f"  {name.ljust(width)}{summaries[name]}" for name in names)
        for title, names in sections.items()
    ]
    parser.epilog = (
        "\n\n".join(listing)
        + "\n\nStart with `homunculus demo start`, then `homunculus describe`. Every command has --help."
    )
    return parser


def _hint(args: argparse.Namespace, error: RobotError) -> str | None:
    """What to do about a refusal. Three of them depend on more than the error's code."""
    if args.run is stop and error.code in ("session_invalid", "stale_context"):
        # Whoever typed a stop wants the robot stopped. A session that has ended, or that belongs to an
        # earlier context, stops nothing, so the advice is the operator's stop and not a new session.
        return STOP_WITHOUT_SESSION
    if error.code == "request_conflict":
        # Whoever repeats an ID wants to know what its request did; under a new ID it would run again.
        inspect = INSPECT.get(error.details.get("kind"))
        operation_id = error.details["operation_id"]
        shown = f"`homunculus {inspect} {operation_id}` shows what that request did; " if inspect else ""
        return f"{shown}a request that differs in any way (arguments, session, or the robot's context) needs a new ID"
    flag = DEADLINE_FLAGS.get(error.details.get("parameter"))
    if error.code == "limit_exceeded" and flag and "minimum" in error.details:
        # A movement that cannot fit its deadline. That deadline is often the default, which nobody typed.
        return f"allow more than {error.details['minimum']:g} s with {flag}, or ask for a shorter movement"
    return HINTS.get(error.code)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    out = sys.stdout  # None when the command starts with standard output closed; print() then writes nothing
    if isinstance(out, io.TextIOWrapper):
        # A character the terminal's encoding cannot show is written escaped, as on standard error, not raised.
        out.reconfigure(errors="backslashreplace")
    _Parser.json_errors = "--json" in arguments
    args = _parser().parse_args(arguments)

    def fail(code: str, message: str, status: int, hint: str | None = None, details: Any = None) -> int:
        if args.json:
            print(
                json.dumps({"error": {"code": code, "message": message, "hint": hint, "details": details or {}}}),
                file=sys.stderr,
            )
        else:
            shown = f"{message} [{code}]" if status == REFUSED else message
            print(f"error: {shown}" + (f"\n  hint: {hint}" if hint else ""), file=sys.stderr)
        return status

    try:
        status = args.run(args)
        if out is not None:
            out.flush()  # a reader that went away is noticed here, not as noise when the interpreter exits
        return status
    except UsageError as error:
        return fail("usage", str(error), USAGE)
    except ValidationError as error:
        first = error.errors()[0]
        return fail("usage", f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}", USAGE)
    except RobotError as error:
        return fail(error.code, error.message, REFUSED, _hint(args, error), error.details)
    except WaitTimeout as error:
        return fail("wait_timeout", error.message, FAILED, error.hint)
    except (WaitInterrupted, RequestInterrupted) as error:
        # The robot or the service goes on without this command, so the command says what it leaves behind.
        return fail("interrupted", error.message, INTERRUPTED, error.hint)
    except ClientError as error:
        return fail("client", error.message, NO_SERVICE, error.hint)
    except StartupError as error:  # `homunculus serve`: the service or its backend says why it cannot start
        return fail("startup", str(error), FAILED)
    except BrokenPipeError:
        # The reader of the output went away, as in `homunculus events --follow | head`. Standard output is pointed
        # elsewhere so that nothing more is written to the closed pipe, and the command ends as shell tools do.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return OUTPUT_CLOSED
    except OSError as error:  # a file or directory this command needs, such as $HOMUNCULUS_HOME or --save
        return fail("client", _file_problem(error), NO_SERVICE)
    except KeyboardInterrupt:
        return INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
