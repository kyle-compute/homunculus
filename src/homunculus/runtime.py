"""The local robot runtime: sole motion authority for one robot.

It owns the client control session, finite goals, target streams, retained
controller targets, deadlines, and stopping decisions. Request handlers call
the public methods, which only record decisions. tick() alone commands the
backend, so ownership, conditions, and dispatch are serialized on one control
thread and a stop is always dispatched ahead of any other command.

Three lifetimes stay separate: the session (who may command), a goal or stream
(one movement), and what a resource's controller keeps holding afterwards.

Reading order: the backend boundary, then Runtime with its control tick first
and one section per lifetime after it.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, NoReturn, Protocol, Self

from pydantic import ConfigDict, Field

from .history import History
from .journal import Entry, Journal, digest
from .models import (
    TERMINAL_GOAL_STATES,
    Action,
    Authority,
    BackendInfo,
    BaseVelocityLimits,
    BaseVelocityTarget,
    CameraImage,
    ClockInfo,
    Contract,
    ControlLoop,
    Event,
    Goal,
    GoalResult,
    GoalSpec,
    GoalSubmit,
    ImageHandle,
    ManagementRequest,
    Observation,
    Operation,
    ProfileBinding,
    ResourceStatus,
    Retained,
    RobotDescription,
    RobotError,
    Sample,
    SensorInfo,
    ServiceLimits,
    Session,
    SessionOpen,
    SessionRenew,
    SessionStatus,
    Stream,
    StreamOpen,
    StreamUpdate,
    StreamUpdateResult,
    TargetAck,
    Unavailable,
    ValidateRequest,
    Validation,
)
from .profiles import (
    PROFILES,
    ActivityStatus,
    JointReference,
    MotionJudge,
    Reading,
    check_velocity,
    find_binding,
    resolve_stream,
)

log = logging.getLogger(__name__)

# What every resource sample carries in its values, keyed by joint (or, for a base, pose component) name.
MAPS = ("positions", "velocities")
NO_CHANNEL = Unavailable(reason="not implemented in this prototype; send updates with PUT on a kept-alive connection")


# --------------------------------------------------------------------------
# The backend boundary. It lives beside its only consumer.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """An authenticated caller. IDs are not credentials; this is."""

    name: str
    operator: bool = False


class Clock(Protocol):
    def now(self) -> float:
        """Monotonic host seconds."""


class SystemClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class ResourceSpec:
    """A control group: joints that one controller owns together."""

    id: str
    role: str
    joints: list[str]
    controller: str


@dataclass
class BackendFacts:
    """What a backend declares about itself. Re-read after a management operation."""

    info: BackendInfo
    world_id: str | None
    mode: str
    clock_mode: Literal["realtime", "stepped"]
    clocks: list[ClockInfo]
    frames: list[str]
    sensors: list[SensorInfo]
    revisions: dict[str, str]
    resources: list[ResourceSpec]
    bindings: list[ProfileBinding]
    # Groups of resources whose controllers are validated to run at the same time.
    concurrent: list[list[str]]
    management: list[str]
    period_s: float


@dataclass
class Feedback:
    execution_time_s: float
    # One sample per resource, keyed by resource id, with "positions" and
    # "velocities" maps; a backend may add other components, and samples for
    # advertised sensors that are not resources.
    samples: dict[str, Sample]
    # Per goal or stream: what its judge (see profiles.py) decided from these measurements.
    activities: dict[str, ActivityStatus] = field(default_factory=dict)
    # Per resource: measured standstill under the fallback.
    stopped: dict[str, bool] = field(default_factory=dict)
    # Per resource: the target its controller maintains while no activity runs.
    holding: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per resource: controller's current joint reference for smooth mid-motion handoffs.
    references: dict[str, JointReference] = field(default_factory=dict)
    # Host time the backend did not turn into execution time, in total for the current world.
    lag_s: float | None = None


@dataclass
class Frame:
    """A recorded image and its bytes. The runtime stores the bytes without decoding them."""

    image: CameraImage
    data: bytes


class StartupError(Exception):
    """A robot or service cannot be served as configured. The message says why, for the person who started it."""


class GoalCommand(Contract):
    """The resolved contract and the runtime's evaluator for this invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    judge: MotionJudge = Field(exclude=True, repr=False)


class Backend(ABC):
    """What the runtime needs from a robot or simulator. Subclass it to add your own.

    Implement the four abstract methods. The others are needed only for what describe() advertises: track() for
    a binding with a stream mode, capture() for a camera sensor, manage() for each management operation. The
    runtime never calls them for anything not advertised, so their defaults only report a backend that
    advertises more than it implements.

    Every method except capture() is called only from the control thread, inside tick(), and must return
    promptly. Backends reporting readings at physics or device cadence feed the runtime-owned
    judge on the command they receive. Other backends are judged on fresh samples from advance().
    """

    reports_goal_samples: ClassVar[bool] = False

    @abstractmethod
    def describe(self) -> BackendFacts:
        """Declare resources, profile bindings, sensors, clocks, and the control period."""

    @abstractmethod
    def apply(self, activity_id: str, contract: Contract) -> None:
        """Start a finite goal on its resources, replacing what they retained."""

    @abstractmethod
    def stop(self, resources: Sequence[str]) -> None:
        """Apply the mode's fallback: end travel, hold the measured position, keep control."""

    @abstractmethod
    def advance(self) -> Feedback:
        """Step or poll the robot; report measurements and progress."""

    def track(self, activity_id: str, target: str, setpoint: BaseVelocityTarget, limits: BaseVelocityLimits) -> None:
        """Set a stream's newest setpoint; no stop between updates."""
        raise NotImplementedError(f"{type(self).__name__} advertises a stream but does not implement track()")

    def manage(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any] | Future[dict[str, Any]]:
        """Perform an advertised management operation, or return its pending result without blocking."""
        raise NotImplementedError(f"{type(self).__name__} advertises {operation} but does not implement manage()")

    def capture(self, sensor: str) -> list[Frame]:
        """Record frames from an advertised camera or spectator camera. Called from request threads.

        Convert what the source gives into CameraImage's conventions here, at the boundary: undistorted
        pixels, the lens in pixels, and the camera's pose with optical axes. A camera whose SensorInfo has
        video_fps is also captured for its live video, that many times a second while someone watches, and
        then returns one rgb frame in a format browsers show, such as JPEG.
        """
        raise NotImplementedError(f"{type(self).__name__} advertises camera {sensor} but does not implement capture()")

    def close(self) -> None:  # noqa: B027 - optional: most backends hold nothing that outlives them
        """Release what the backend holds: processes, connections, files. Nothing by default.

        The code that builds a backend closes it, by building it in a `with`. The runtime does not.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# Runtime state. The public models are the state; these add private bookkeeping.
# --------------------------------------------------------------------------


@dataclass
class _Session:
    public: Session
    expires_at: float


@dataclass
class _Resource:
    status: ResourceStatus
    stop_deadline_wall_s: float
    # While stopping or recovering: when confirmation is due, and the backend
    # reading during which the command was sent.
    deadline: float | None = None
    commanded_at: int | None = None


@dataclass
class _Ending:
    """How a goal will end once its stop is confirmed."""

    state: str
    reason: str
    detail: str | None = None
    measurements: dict[str, Any] = field(default_factory=dict)


@dataclass
class _GoalEvaluation:
    judge: MotionJudge
    started_s: float
    last_sample_s: float


@dataclass
class _GoalRecord:
    goal: Goal
    wall_deadline: float
    execution_deadline: float | None
    ending: _Ending | None = None
    dispatch_write: Future[None] | None = None
    evaluation: _GoalEvaluation | None = None


@dataclass
class _StreamRecord:
    stream: Stream
    expires_at: float
    basis_sequence: int
    # Latest-value mailbox: the newest accepted target the controller has not applied.
    mailbox: BaseVelocityTarget | None
    # Recent acknowledgements by sequence, with the digest of the update that earned each.
    acks: OrderedDict[int, tuple[str, TargetAck]]
    ending: str | None = None
    dispatch_write: Future[None] | None = None


@dataclass
class _Management:
    operation: Operation
    result: Future[dict[str, Any]]
    deadline: float | None


def _ns(seconds: float) -> str:
    return str(round(seconds * 1e9))


def _new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(6)}"


class Runtime:
    def __init__(
        self,
        robot_id: str,
        backend: Backend,
        journal: Journal,
        *,
        boot_id: str,
        clock: Clock | None = None,
        limits: ServiceLimits | None = None,
    ):
        self.robot_id = robot_id
        self.boot_id = boot_id
        self._backend = backend
        self._journal = journal
        self._clock = clock or SystemClock()
        self._limits = (limits or ServiceLimits()).model_copy(update={"journal_quota": journal.quota})
        self._history = History(robot_id, boot_id, self._limits)
        self._lock = threading.RLock()
        self._admission = threading.Lock()
        self._facts = backend.describe()
        unregistered = sorted({binding.profile for binding in self._facts.bindings} - PROFILES.keys())
        if unregistered:
            raise StartupError(
                f"{type(backend).__name__} binds {', '.join(unregistered)}, which no one registered; pass each "
                "profile to homunculus.profiles.register() before the service starts"
            )
        self._epoch = 0
        self.context_id = self._context(0)

        deadlines = {spec.id: 0.0 for spec in self._facts.resources}
        for binding in self._facts.bindings:
            for rid in binding.resources:
                deadlines[rid] = max(deadlines[rid], binding.stop_deadline_wall_s)
        self._resources = {
            spec.id: _Resource(
                ResourceStatus(
                    id=spec.id, role=spec.role, joints=spec.joints, controller=spec.controller, condition="ready"
                ),
                deadlines[spec.id],
            )
            for spec in self._facts.resources
        }
        self._stop_queue: set[str] = set()
        self._stop_generation = 0

        self._session: _Session | None = None
        self._last_session: Session | None = None
        self._goals: dict[str, _GoalRecord] = {}
        self._streams: dict[str, _StreamRecord] = {}
        self._operations: dict[str, Operation] = {}
        self._awaited: dict[str, list[str]] = {}  # operation id -> resources it waits on
        self._management: deque[tuple[Operation, ManagementRequest]] = deque()
        self._transition_in_progress: _Management | None = None

        self._ticks = 0
        self._last_tick: float | None = None
        self._worst_interval = 0.0
        # Counts backend readings, so "newer than" never depends on clock resolution.
        self._state_sequence = 0
        self._feedback_time = self._clock.now()
        self._feedback = Feedback(execution_time_s=0.0, samples={})
        self._feedback = self._read_backend()
        # Activation and start conditions read these maps; a backend that leaves them out fails here, at start.
        for spec in self._facts.resources:
            sample = self._feedback.samples.get(spec.id)  # none yet only if the backend could not be read
            if sample is not None and not all(isinstance(sample.values.get(key), dict) for key in MAPS):
                raise StartupError(
                    f"{type(backend).__name__} reports no {' and '.join(MAPS)} maps for resource {spec.id!r}; "
                    "every resource sample needs them"
                )
        self._set_holding(self._resources.values())
        self._interrupt_previous_boot()

    @property
    def period_s(self) -> float:
        return self._facts.period_s

    # ------------------------------------------------------------------
    # The control tick: the only place the backend is commanded
    # ------------------------------------------------------------------

    def tick(self) -> None:
        with self._lock:
            now = self._clock.now()
            if self._last_tick is not None:
                self._worst_interval = max(self._worst_interval, now - self._last_tick)
            self._last_tick = now
            self._ticks += 1

            self._expire(now)
            self._dispatch_stops()
            self._feedback = self._read_backend()
            self._feedback_time = now
            self._state_sequence += 1

            self._settle_resources(now)
            self._progress_goals(now)
            self._progress_streams(now)
            self._progress_operations(now)
            self._run_management(now)

            # Dispatch. Stopping goes first, ahead of any new target or activation.
            self._dispatch_stops()
            self._apply_targets(now)
            for record in list(self._goals.values()):
                if record.goal.state == "accepted":
                    self._activate(record, now)

            # Memory is bounded here and nowhere else. Ending a goal or stream never removes a record, so
            # the loops that end them (a stop, a session ending, a barrier) can walk these tables safely.
            referenced = {
                ref
                for r in self._goals.values()
                if r.goal.state == "accepted"
                for ref in (r.goal.contract.after_goal_id, r.goal.contract.replace_goal_id)
                if ref is not None
            }
            self._trim(self._goals, lambda r: r.goal.state in TERMINAL_GOAL_STATES and r.goal.goal_id not in referenced)
            self._trim(self._streams, lambda r: r.stream.state == "closed")
            self._trim(self._operations, lambda op: op.state != "running")

    def close(self) -> None:
        """Leave the robot under its fallback. Unfinished work is interrupted on the next boot.

        The backend stays open for whoever built it to close.
        """
        with self._lock:
            try:
                self._backend.stop(list(self._resources))
            except Exception:  # the other robots of the service are still to be stopped
                log.exception("backend stop failed during shutdown")

    def _read_backend(self) -> Feedback:
        try:
            return self._backend.advance()
        except Exception:
            # No feedback is not standstill: every sample becomes invalid and nothing is confirmed.
            log.exception("backend advance failed; treating all feedback as lost")
            lost = {rid: sample.model_copy(update={"valid": False}) for rid, sample in self._feedback.samples.items()}
            return Feedback(execution_time_s=self._feedback.execution_time_s, samples=lost)

    def _expire(self, now: float) -> None:
        """Run before every admission and dispatch, so nothing expired can be extended or applied."""
        if self._session is not None and now >= self._session.expires_at:
            self._end_session("expired", "lifetime_elapsed", now)
        for record in self._streams.values():
            if record.stream.state == "open" and now >= record.expires_at:
                self._end_stream(record, "expired", now, "no fresh target arrived before the last one expired")

    def _valid(self, rid: str) -> bool:
        sample = self._feedback.samples.get(rid)
        return sample is not None and sample.valid

    def _require_measured(self, resources: Sequence[str]) -> None:
        """Nothing is admitted, goal or stream, for a resource whose state is not being measured."""
        for rid in resources:
            if not self._valid(rid):
                raise RobotError("start_condition_failed", f"there is no valid measurement of {rid!r}", resource=rid)

    # ------------------------------------------------------------------
    # Describe and observe
    # ------------------------------------------------------------------

    def describe(self) -> RobotDescription:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            facts, session = self._facts, self._session
            host_clock = ClockInfo(
                id="host_monotonic",
                epoch=self.boot_id,
                description="runtime clock for issue times, sessions, wall deadlines, and stream freshness",
            )
            return RobotDescription(
                robot_id=self.robot_id,
                boot_id=self.boot_id,
                context_id=self.context_id,
                world_id=facts.world_id,
                backend=facts.info,
                revisions=facts.revisions,
                mode=facts.mode,
                clock_mode=facts.clock_mode,
                clocks=[host_clock, *facts.clocks],
                frames=facts.frames,
                sensors=facts.sensors,
                resources=self._resource_statuses(),
                profiles=[self._availability(binding) for binding in facts.bindings],
                concurrent_groups=facts.concurrent,
                management=facts.management,
                session=SessionStatus(
                    held=session is not None,
                    principal=session.public.principal if session else None,
                    expires_in_s=max(0.0, session.expires_at - now) if session else None,
                ),
                limits=self._limits,
                persistent_channel=NO_CHANNEL,
                control_loop=ControlLoop(
                    period_s=facts.period_s,
                    ticks=self._ticks,
                    worst_interval_s=self._worst_interval,
                    execution_lag_s=self._feedback.lag_s,
                ),
            )

    def observe(self, cameras: Sequence[str] = (), after_sequence: int | None = None) -> Observation:
        """Issue a new immutable observation of the latest measurements.

        With after_sequence, refuse (no_new_observation) until the backend has
        been read again since that observation was issued.
        """
        with self._lock:
            for camera in cameras:
                self._camera(camera)
            if after_sequence is not None:
                earlier = self._history.with_sequence(after_sequence)
                if earlier is not None and earlier.state_sequence == self._state_sequence:
                    raise RobotError(
                        "no_new_observation",
                        f"no measurement newer than observation {after_sequence} yet",
                        after_sequence=after_sequence,
                    )
        # Rendering can be slow, so it runs outside the lock: stopping never waits on a camera.
        captured = [(camera, frame) for camera in cameras for frame in self._backend.capture(camera)]
        with self._lock:
            images = [
                ImageHandle(
                    **dict(frame.image),
                    handle=self._history.add_image(frame.image.media_type, frame.data),
                    sensor=camera,
                )
                for camera, frame in captured
            ]
            return self._issue(self._clock.now(), images).model_copy(deep=True)

    def observation(self, observation_id: str) -> Observation:
        with self._lock:
            return self._history.observation(observation_id).model_copy(deep=True)

    def image(self, handle: str) -> tuple[str, bytes]:
        with self._lock:
            return self._history.image(handle)

    def video_fps(self, camera: str) -> float:
        """The frames per second of a camera's live video. Refused for a camera that takes snapshots only."""
        with self._lock:
            sensor = self._camera(camera, kinds=("camera", "spectator_camera"))
        if sensor.video_fps is None:
            raise RobotError(
                "unsupported_sensor",
                f"camera {camera!r} takes snapshots only, not live video: {sensor.description}",
                camera=camera,
            )
        return sensor.video_fps

    def video_frame(self, camera: str) -> Frame:
        """A new frame for a camera's live video. Unlike an observation's images, it is not recorded."""
        self.video_fps(camera)  # asked for each frame: a management operation may have changed the cameras
        color = [frame for frame in self._backend.capture(camera) if frame.image.kind == "rgb"]
        if len(color) != 1:
            raise RobotError(
                "backend_error", f"camera {camera!r} returned {len(color)} rgb frames; a live video needs one"
            )
        return color[0]

    def _camera(self, name: str, kinds: tuple[str, ...] = ("camera",)) -> SensorInfo:
        """An advertised sensor of one of these kinds, or the refusal that lists the ones there are. Called with
        the lock held. Observations record the robot's own cameras only; live video also plays spectator ones."""
        cameras = {sensor.id: sensor for sensor in self._facts.sensors if sensor.kind in kinds}
        if name in cameras:
            return cameras[name]
        if any(sensor.id == name and sensor.kind == "spectator_camera" for sensor in self._facts.sensors):
            message = (
                f"camera {name!r} is for people watching its live video, not one of the robot's cameras "
                f"({', '.join(sorted(cameras)) or 'it has none'})"
            )
        else:
            message = f"this robot has no camera named {name!r}"
        raise RobotError("unsupported_sensor", message, cameras=sorted(cameras))

    def events(self, after: int) -> tuple[list[Event], bool]:
        with self._lock:
            return self._history.events(after)

    def _resource_statuses(self) -> list[ResourceStatus]:
        """Snapshot every resource with its derived queue of accepted-but-waiting goals."""
        queued: dict[str, list[str]] = {rid: [] for rid in self._resources}
        for record in self._goals.values():
            goal = record.goal
            if goal.state == "accepted":
                for rid in goal.contract.resources:
                    if self._resources[rid].status.activity != goal.goal_id:
                        queued[rid].append(goal.goal_id)
        return [r.status.model_copy(deep=True, update={"queued": queued[rid]}) for rid, r in self._resources.items()]

    def _issue(self, now: float, images: Sequence[ImageHandle] = ()) -> Observation:
        return self._history.issue(
            context_id=self.context_id,
            now=now,
            measured_at=self._feedback_time,
            state_sequence=self._state_sequence,
            samples=self._feedback.samples,
            resources=self._resource_statuses(),
            images=images,
        )

    # ------------------------------------------------------------------
    # Sessions: who may command
    # ------------------------------------------------------------------

    def open_session(self, principal: Principal, request: SessionOpen) -> Session:
        body = request.model_dump(mode="json")
        with self._admission:
            prior = self._replay(principal, request.request_id, body, "session")
            if prior is not None:
                with self._lock:
                    now = self._clock.now()
                    self._expire(now)
                    live = self._session_view(prior.operation_id, now)
                if live is not None:
                    return live
                return self._ended_session(prior.operation_id) or Session.model_validate(prior.document)
            with self._lock:
                now = self._clock.now()
                self._expire(now)
                self._check_context(request.context_id)
                lifetime = self._lifetime(request.lifetime_wall_s, self._limits.default_session_lifetime_wall_s)
                if request.takeover:
                    if not principal.operator:
                        raise RobotError("permission_denied", "takeover needs operator permission")
                    # Revoke first. The old owner's activity must finish stopping before anyone else may command.
                    self._end_session("revoked", "operator_takeover", now)
                if self._session is not None:
                    holder = self._session
                    raise RobotError(
                        "session_conflict",
                        f"principal {holder.public.principal!r} holds the control session for this robot",
                        principal=holder.public.principal,
                        expires_in_s=max(0.0, holder.expires_at - now),
                    )
                stopping = [rid for rid, r in self._resources.items() if r.status.condition == "stopping"]
                if stopping:
                    raise RobotError(
                        "resource_busy",
                        "the previous owner's activity is still stopping; open the session once that is confirmed",
                        resources=stopping,
                    )
                session = Session(
                    session_id=_new_id("ses"),
                    robot_id=self.robot_id,
                    context_id=self.context_id,
                    principal=principal.name,
                    state="active",
                    lifetime_wall_s=lifetime,
                    expires_at_ns=_ns(now + lifetime),
                    expires_in_s=lifetime,
                )
            self._admit(
                principal, "session", session.session_id, request_id=request.request_id, request=body, document=session
            )
            with self._lock:
                self._session = _Session(session, now + lifetime)
                now = self._clock.now()
                if session.context_id != self.context_id:
                    self._end_session("revoked", "stale_context", now)
                    return session.model_copy()
                self._expire(now)
                if self._session is None:
                    return session.model_copy()
                self._history.emit("session.opened", now, session_id=session.session_id, principal=principal.name)
                return self._session_view(session.session_id, now)

    def renew_session(self, principal: Principal, session_id: str, request: SessionRenew) -> Session:
        """Extend the session only. Goal deadlines and stream targets are untouched."""
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            session = self._own_session(principal, session_id)
            lifetime = self._lifetime(request.lifetime_wall_s, session.public.lifetime_wall_s)
            session.public.lifetime_wall_s = lifetime
            session.expires_at = now + lifetime
            self._history.emit("session.renewed", now, session_id=session_id)
            return self._session_view(session_id, now)

    def close_session(self, principal: Principal, session_id: str) -> Session:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            if self._session is not None and self._session.public.session_id == session_id:
                session = self._own_session(principal, session_id)
                self._end_session("closed", "client_closed", now)
                return session.public.model_copy()
        ended = self._ended_session(session_id)
        if ended is not None and ended.principal == principal.name:
            return ended  # closing it again changes nothing
        return self._invalid_session(session_id, ended)

    def _lifetime(self, requested: float | None, default: float) -> float:
        lifetime = requested or default
        maximum = self._limits.max_session_lifetime_wall_s
        if lifetime > maximum:
            raise RobotError(
                "limit_exceeded",
                f"lifetime_wall_s={lifetime:g} exceeds the maximum {maximum:g}",
                parameter="lifetime_wall_s",
                maximum=maximum,
            )
        return lifetime

    def _session_view(self, session_id: str, now: float) -> Session | None:
        """The live session with its remaining lifetime, or None when session_id is not the live one."""
        session = self._session
        if session is None or session.public.session_id != session_id:
            return None
        return session.public.model_copy(
            update={"expires_at_ns": _ns(session.expires_at), "expires_in_s": max(0.0, session.expires_at - now)}
        )

    def _ended_session(self, session_id: str | None) -> Session | None:
        """How a session that is over ended. The journal is its only record, so this holds in any boot."""
        if session_id is None or not session_id.startswith("ses-"):
            return None
        outcome = self._journal.outcome(self.robot_id, session_id)
        if outcome is not None:
            return Session.model_validate(outcome)
        entry = self._journal.find(self.robot_id, session_id)
        if entry is None or entry.kind != "session" or not entry.finished:
            return None
        return Session.model_validate(entry.document)

    def _authority(self, principal: Principal, session_id: str | None, context_id: str | None) -> None:
        """Every motion mutation passes here: current context, live session, same principal."""
        self._check_context(context_id)
        self._own_session(principal, session_id)

    def _may_end(self, principal: Principal, authority: Authority) -> bool:
        """Whether this caller ends motion as an operator, outside any session.

        Only an operator may, and only by naming no session. A session that is named is checked like
        anyone's: an operator whose own session has ended does not stop the next owner with it by accident.
        """
        if principal.operator and authority.session_id is None:
            return True
        self._authority(principal, authority.session_id, authority.context_id)
        return False

    def _check_context(self, context_id: str | None) -> None:
        if context_id != self.context_id:
            raise RobotError(
                "stale_context",
                "the robot's execution context changed (restart, reset, recovery, or mode change); "
                "read the description again and open a new session",
                current_context_id=self.context_id,
                boot_id=self.boot_id,
            )

    def _own_session(self, principal: Principal, session_id: str | None) -> _Session:
        session = self._session
        if session is None or session.public.session_id != session_id:
            ended = self._last_session
            self._invalid_session(session_id, ended if ended is not None and ended.session_id == session_id else None)
        if session.public.principal != principal.name:
            raise RobotError("permission_denied", "this session belongs to another principal")
        return session

    @staticmethod
    def _invalid_session(session_id: str | None, ended: Session | None) -> NoReturn:
        state = ended.state if ended else "unknown"
        raise RobotError(
            "session_invalid",
            f"session {session_id} is {state}; open a new session to command this robot",
            session_id=session_id,
            state=state,
            reason=ended.reason if ended else None,
        )

    def _end_session(self, state: str, reason: str, now: float) -> None:
        """Revoke authority and stop what it commanded. Retained holds stay in place."""
        session = self._session
        if session is None:
            return
        self._session = None
        public = session.public
        public.state, public.reason, public.expires_in_s = state, reason, 0.0
        public.expires_at_ns = _ns(min(session.expires_at, now))
        self._last_session = public
        goal_state, goal_reason = {
            "closed": ("cancelled", "session_closed"),
            "expired": ("failed", "session_expired"),
        }.get(state, ("cancelled", reason))
        for record in self._goals.values():
            if record.goal.session_id == public.session_id and record.goal.state not in TERMINAL_GOAL_STATES:
                self._end_goal(record, _Ending(goal_state, goal_reason), now, cascade=False)
        for stream in self._streams.values():
            if stream.stream.session_id == public.session_id and stream.stream.state == "open":
                self._end_stream(stream, "interrupted", now, goal_reason)
        # Written after the stopping above is under way. The journal is the one record of a session that is
        # over: a repeated open or close reads it there, in this boot or a later one.
        self._journal.record_later(self.robot_id, public.session_id, public.model_dump(mode="json"), finished=True)
        self._history.emit(f"session.{state}", now, session_id=public.session_id, reason=reason)

    def _context(self, epoch: int) -> str:
        seed = f"{self.boot_id}:{self.robot_id}:{epoch}".encode()
        return "ctx-" + hashlib.sha256(seed).hexdigest()[:12]

    def _change_context(self, why: str, now: float) -> None:
        """Start a new execution context. Authority granted in the old one ends with it."""
        previous = self.context_id
        self._epoch += 1
        self.context_id = self._context(self._epoch)
        self._end_session("revoked", why, now)
        self._history.emit("context.changed", now, previous=previous, current=self.context_id, reason=why)

    # ------------------------------------------------------------------
    # Finite goals: accepted -> executing -> finishing -> succeeded,
    # or -> stopping -> cancelled / failed, or interrupted by restart or reset
    # ------------------------------------------------------------------

    def validate(self, request: ValidateRequest) -> Validation:
        """Check a proposed goal or stream as it would be submitted. Nothing is reserved or dispatched."""
        with self._lock:
            if (request.goal is None) == (request.stream is None):
                raise RobotError("invalid_request", "give exactly one of goal or stream")
            if request.goal is not None:
                now = self._clock.now()
                contract, binding = self._contract(request.goal)
                if request.goal.replace_goal_id is not None:
                    target, _ = self._replace_target(request.goal.replace_goal_id, contract, binding)
                    if target.state == "accepted":
                        parent = (
                            self._predecessor(target.contract.after_goal_id)
                            if target.contract.after_goal_id is not None
                            else None
                        )
                        self._check_start(contract, now, after=parent)
                    else:
                        reference = self._feedback.references.get(contract.resources[0])
                        self._check_start(contract, now, reference=reference)
                else:
                    predecessor = self._predecessor(request.goal.after_goal_id) if request.goal.after_goal_id else None
                    self._check_start(contract, now, after=predecessor)
                return Validation(resources=contract.resources, contract=contract)
            binding = find_binding(self._facts.bindings, request.stream.profile, request.stream.target, "stream")
            self._require_available(binding)
            return Validation(resources=binding.resources, limits=resolve_stream(request.stream, binding))

    def submit_goal(self, principal: Principal, request: GoalSubmit) -> Goal:
        body = request.model_dump(mode="json")
        with self._admission:
            prior = self._replay(principal, request.request_id, body, "goal")
            if prior is not None:
                return self.goal(prior.operation_id)
            with self._lock:
                now = self._clock.now()
                self._expire(now)
                self._authority(principal, request.session_id, request.context_id)
                contract, binding = self._contract(request)
                waiting = False
                if request.replace_goal_id is not None:
                    target_goal, _ = self._replace_target(
                        request.replace_goal_id, contract, binding, session_id=request.session_id
                    )
                    ignore = self._descendants(target_goal.goal_id)
                    if target_goal.state == "accepted":
                        parent = (
                            self._predecessor(target_goal.contract.after_goal_id, session_id=request.session_id)
                            if target_goal.contract.after_goal_id is not None
                            else None
                        )
                        contract.after_goal_id = target_goal.contract.after_goal_id
                        waiting = parent is not None and parent.state not in TERMINAL_GOAL_STATES
                        self._check_start(contract, now, after=parent)
                        self._check_free(contract.resources, after=parent, ignore=ignore)
                    else:
                        reference = self._feedback.references.get(contract.resources[0])
                        self._check_start(contract, now, reference=reference)
                        self._check_free(contract.resources, ignore=ignore)
                else:
                    predecessor = (
                        self._predecessor(request.after_goal_id, session_id=request.session_id)
                        if request.after_goal_id is not None
                        else None
                    )
                    waiting = predecessor is not None and predecessor.state not in TERMINAL_GOAL_STATES
                    if waiting:
                        queued_count = sum(
                            1
                            for r in self._goals.values()
                            if r.goal.state == "accepted"
                            and r.goal.goal_id != self._resources[r.goal.contract.resources[0]].status.activity
                        )
                        if queued_count >= self._limits.max_queued_goals:
                            raise RobotError(
                                "limit_exceeded",
                                f"at most {self._limits.max_queued_goals} goals may be queued at once",
                                parameter="max_queued_goals",
                                maximum=self._limits.max_queued_goals,
                            )
                    self._check_start(contract, now, after=predecessor)
                    self._check_free(contract.resources, after=predecessor)
                goal = Goal(
                    goal_id=_new_id("goal"),
                    request_id=request.request_id,
                    robot_id=self.robot_id,
                    context_id=self.context_id,
                    session_id=request.session_id,
                    principal=principal.name,
                    state="accepted",
                    contract=contract,
                    accepted_time_ns=_ns(now),
                )
                # Identity and the resolved command are durable before any dispatch.
                record = _GoalRecord(
                    goal,
                    # Wall deadline always starts at admission; execution deadline starts when head-of-line.
                    wall_deadline=now + contract.max_wall_s,
                    execution_deadline=None if waiting else self._feedback.execution_time_s + contract.max_execution_s,
                )
                stop_generation = self._stop_generation
            self._admit(principal, "goal", goal.goal_id, request_id=request.request_id, request=body, document=goal)
            with self._lock:
                now = self._clock.now()
                self._expire(now)
                self._goals[goal.goal_id] = record
                if reason := self._admission_interrupted(goal.context_id, stop_generation, goal.session_id):
                    self._finish_goal(record, "cancelled", reason, now)
                    return goal.model_copy(deep=True)
                claim = True
                if request.replace_goal_id is not None:
                    found_target = self._find_goal(request.replace_goal_id)
                    if found_target is None:
                        self._finish_goal(record, "cancelled", "predecessor_failed", now)
                        return goal.model_copy(deep=True)
                    t_goal, t_record = found_target
                    if t_goal.state == "accepted" and t_record is not None:
                        contract.after_goal_id = t_goal.contract.after_goal_id
                        for rid in t_goal.contract.resources:
                            if self._resources[rid].status.activity == t_goal.goal_id:
                                self._resources[rid].status.activity = None
                        self._finish_goal(
                            t_record,
                            "cancelled",
                            "superseded",
                            now,
                            detail=f"superseded by {goal.goal_id}",
                            measurements={"superseded_by": goal.goal_id},
                        )
                        self._cancel_dependents(
                            t_goal.goal_id, "cancelled", "superseded", now, exclude=frozenset({goal.goal_id})
                        )
                    elif t_goal.state in ("executing", "finishing"):
                        if not binding.replaceable or contract.resources != t_goal.contract.resources:
                            self._finish_goal(
                                record,
                                "cancelled",
                                "target_already_executing",
                                now,
                                detail=f"goal {t_goal.goal_id} began executing before replacement was admitted",
                            )
                            return goal.model_copy(deep=True)
                        contract.after_goal_id = None
                        if record.execution_deadline is None:
                            record.execution_deadline = self._feedback.execution_time_s + contract.max_execution_s
                        claim = False
                    elif t_goal.state == "succeeded":
                        contract.after_goal_id = None
                        if record.execution_deadline is None:
                            record.execution_deadline = self._feedback.execution_time_s + contract.max_execution_s
                        claim = all(
                            self._resources[rid].status.activity is None
                            and self._resources[rid].status.condition == "ready"
                            for rid in contract.resources
                        )
                    else:
                        self._finish_goal(
                            record,
                            "cancelled",
                            "predecessor_failed",
                            now,
                            detail=f"target goal {request.replace_goal_id} ended {t_goal.state}",
                            measurements={
                                "predecessor_goal_id": request.replace_goal_id,
                                "predecessor_state": t_goal.state,
                                "predecessor_reason": self._goal_reason(t_goal, t_record),
                            },
                        )
                        return goal.model_copy(deep=True)
                if contract.after_goal_id is not None:
                    found = self._find_goal(contract.after_goal_id)
                    if found is not None:
                        pred_goal, pred_record = found
                        if pred_goal.state in ("stopping", "failed", "cancelled", "interrupted"):
                            pred_reason = self._goal_reason(pred_goal, pred_record)
                            self._finish_goal(
                                record,
                                "cancelled",
                                "predecessor_failed",
                                now,
                                detail=f"predecessor goal {contract.after_goal_id} ended {pred_goal.state}",
                                measurements={
                                    "predecessor_goal_id": contract.after_goal_id,
                                    "predecessor_state": pred_goal.state,
                                    "predecessor_reason": pred_reason,
                                },
                            )
                            return goal.model_copy(deep=True)
                        if pred_goal.state == "succeeded":
                            if record.execution_deadline is None:
                                record.execution_deadline = self._feedback.execution_time_s + contract.max_execution_s
                            claim = all(
                                self._resources[rid].status.activity is None
                                and self._resources[rid].status.condition == "ready"
                                for rid in contract.resources
                            )
                        else:
                            claim = False
                if claim:
                    for rid in contract.resources:
                        self._resources[rid].status.activity = goal.goal_id
                self._history.emit("goal.accepted", now, goal_id=goal.goal_id, profile=contract.action.profile)
                return goal.model_copy(deep=True)

    def goal(self, goal_id: str) -> Goal:
        with self._lock:
            record = self._goals.get(goal_id)
            if record is not None:
                return record.goal.model_copy(deep=True)
        return Goal.model_validate(self._stored(goal_id, "goal"))

    def cancel_goal(self, principal: Principal, goal_id: str, authority: Authority) -> Goal:
        """Stop this goal only. Cancelling a finished goal changes nothing."""
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            record = self._goals.get(goal_id)
            if record is not None:
                if record.goal.state not in TERMINAL_GOAL_STATES:
                    if not self._may_end(principal, authority) and record.goal.session_id != authority.session_id:
                        raise RobotError("permission_denied", "this goal belongs to another session")
                    self._end_goal(record, _Ending("cancelled", "cancelled"), now)
                return record.goal.model_copy(deep=True)
        return Goal.model_validate(self._stored(goal_id, "goal"))

    def _contract(self, goal: GoalSpec) -> tuple[Contract, ProfileBinding]:
        """Resolve a proposed goal against its binding into the concrete command that would run."""
        # A request of an unregistered profile has no target field; no binding offers its profile either.
        binding = find_binding(self._facts.bindings, goal.action.profile, getattr(goal.action, "target", ""), "goal")
        self._require_available(binding)
        deadlines = {
            "max_execution_s": goal.max_execution_s or self._limits.default_max_execution_s,
            "max_wall_s": goal.max_wall_s or self._limits.default_max_wall_s,
        }
        for name, value in deadlines.items():
            if value > self._limits.max_deadline_s:
                raise RobotError(
                    "limit_exceeded",
                    f"{name}={value:g} exceeds the maximum {self._limits.max_deadline_s:g}",
                    parameter=name,
                    maximum=self._limits.max_deadline_s,
                )
        conditions = goal.start_conditions
        if conditions is not None and conditions.max_position_change is not None and conditions.observation_id is None:
            raise RobotError("invalid_request", "max_position_change needs the observation_id it is measured from")
        profile = PROFILES[goal.action.profile]
        action = profile.resolve(goal.action, binding.limits)
        # A goal that states how long its motion takes cannot succeed under a deadline that ends before the
        # motion and its settling do, so it is refused here instead of being moved and then failed. Execution
        # time never runs ahead of a realtime clock; the wall time of a stepped world says nothing about it.
        stated, settling = profile.duration_s(action), binding.limits.settle_time_s
        bounded = ("max_execution_s", "max_wall_s") if self._facts.clock_mode == "realtime" else ("max_execution_s",)
        for name in bounded:
            if stated is not None and stated + settling >= deadlines[name]:
                raise RobotError(
                    "limit_exceeded",
                    f"{name}={deadlines[name]:g} is too short for this request: its motion is stated to take "
                    f"{stated:g} s and must then be measured at rest for {settling:g} s",
                    parameter=name,
                    value=deadlines[name],
                    minimum=stated + settling,
                )
        contract = Contract(
            action=action,
            resources=binding.resources,
            limits=binding.limits,
            stop_deadline_wall_s=binding.stop_deadline_wall_s,
            start_conditions=conditions,
            after_goal_id=goal.after_goal_id,
            replace_goal_id=goal.replace_goal_id,
            **deadlines,
        )
        return contract, binding

    def _find_goal(self, goal_id: str) -> tuple[Goal, _GoalRecord | None] | None:
        record = self._goals.get(goal_id)
        if record is not None:
            return record.goal, record
        if goal_id.startswith("goal-"):
            outcome = self._journal.outcome(self.robot_id, goal_id)
            if outcome is not None:
                return Goal.model_validate(outcome), None
        entry = self._journal.find(self.robot_id, goal_id)
        if entry is None or entry.kind != "goal":
            return None
        return Goal.model_validate(entry.document), None

    @staticmethod
    def _goal_reason(goal: Goal, record: _GoalRecord | None = None) -> str | None:
        if record is not None and record.ending is not None:
            return record.ending.reason
        return goal.result.reason if goal.result is not None else None

    def _ancestor_chain(self, after: Goal | None, *, follow_replace: bool = False) -> list[Goal]:
        """Unfinished ancestors starting at `after`, following `after_goal_id` (and optionally `replace_goal_id`)."""
        chain: list[Goal] = []
        cursor = after
        while cursor is not None and cursor.state not in TERMINAL_GOAL_STATES:
            chain.append(cursor)
            parent_id = cursor.contract.after_goal_id
            if parent_id is None and follow_replace:
                parent_id = cursor.contract.replace_goal_id
            parent = self._goals.get(parent_id) if parent_id is not None else None
            cursor = parent.goal if parent is not None else None
        return chain

    def _descendants(self, goal_id: str, *, include_replace: bool = True) -> set[str]:
        """Non-terminal goals that would be dropped if `goal_id` is replaced (including `goal_id` itself)."""
        dropped = {goal_id}
        for record in self._goals.values():
            goal = record.goal
            if goal.state in TERMINAL_GOAL_STATES:
                continue
            dep = goal.contract.after_goal_id
            if dep is None and include_replace:
                dep = goal.contract.replace_goal_id
            if dep in dropped:
                dropped.add(goal.goal_id)
        return dropped

    def _replace_target(
        self, goal_id: str, contract: Contract, binding: ProfileBinding, *, session_id: str | None = None
    ) -> tuple[Goal, _GoalRecord]:
        found = self._find_goal(goal_id)
        if found is None:
            raise RobotError("start_condition_failed", f"unknown goal {goal_id!r} to replace", replace_goal_id=goal_id)
        target, record = found
        if target.context_id != self.context_id:
            raise RobotError(
                "start_condition_failed",
                f"goal {goal_id!r} belongs to another context",
                replace_goal_id=goal_id,
                context_id=target.context_id,
            )
        if session_id is not None and target.session_id != session_id:
            raise RobotError(
                "permission_denied",
                "cannot replace a goal belonging to another session",
                replace_goal_id=goal_id,
            )
        if target.state not in ("accepted", "executing", "finishing") or record is None:
            raise RobotError(
                "start_condition_failed",
                f"goal {goal_id!r} is {target.state} and cannot be replaced",
                replace_goal_id=goal_id,
                target_state=target.state,
                target_reason=self._goal_reason(target, record),
            )
        for existing in self._goals.values():
            if existing.goal.state == "accepted" and existing.goal.contract.replace_goal_id == goal_id:
                raise RobotError(
                    "resource_busy",
                    f"goal {goal_id!r} already has pending replacement {existing.goal.goal_id}",
                    replace_goal_id=goal_id,
                    activity=existing.goal.goal_id,
                )
        if target.state in ("executing", "finishing"):
            if not binding.replaceable:
                raise RobotError(
                    "unsupported_mode",
                    f"{binding.profile} on {binding.target!r} does not support mid-motion replacement",
                    profile=binding.profile,
                    target=binding.target,
                )
            if contract.resources != target.contract.resources:
                raise RobotError(
                    "invalid_request",
                    f"mid-motion replacement must command the same resources as {goal_id} "
                    f"({target.contract.resources})",
                    replace_goal_id=goal_id,
                    target_resources=target.contract.resources,
                    requested_resources=contract.resources,
                )
        return target, record

    def _predecessor(self, goal_id: str, *, session_id: str | None = None) -> Goal:
        found = self._find_goal(goal_id)
        if found is None:
            raise RobotError("start_condition_failed", f"unknown predecessor goal {goal_id!r}", after_goal_id=goal_id)
        goal, record = found
        if goal.context_id != self.context_id:
            raise RobotError(
                "start_condition_failed",
                f"predecessor goal {goal_id!r} belongs to another context",
                after_goal_id=goal_id,
                context_id=goal.context_id,
            )
        if session_id is not None and goal.session_id != session_id:
            raise RobotError(
                "permission_denied",
                "cannot queue after a goal belonging to another session",
                after_goal_id=goal_id,
            )
        if goal.state in ("stopping", "failed", "cancelled", "interrupted"):
            raise RobotError(
                "start_condition_failed",
                f"predecessor goal {goal_id!r} is {goal.state}",
                after_goal_id=goal_id,
                predecessor_state=goal.state,
                predecessor_reason=self._goal_reason(goal, record),
            )
        return goal

    def _expected_reading(self, rid: str, after: Goal) -> Reading | None:
        """Predicted sample for `rid` after `after` and its unfinished ancestors succeed, or None if unknown."""
        values: Mapping[str, Any] = self._feedback.samples[rid].values
        t = self._feedback.execution_time_s
        for goal in reversed(self._ancestor_chain(after)):
            if rid not in goal.contract.resources:
                continue
            if goal.contract.resources[0] != rid:
                return None
            predicted = PROFILES[goal.contract.action.profile].expected_end(
                goal.contract.action, goal.contract.limits, Reading(values, t)
            )
            if predicted is None:
                return None
            values = predicted
        return Reading(values, t)

    def _check_start(
        self,
        contract: Contract,
        now: float,
        *,
        after: Goal | None = None,
        reference: JointReference | None = None,
    ) -> Action:
        """Run at admission and again at dispatch: nothing starts on unmeasured or changed state.

        Returns the action as it would start now; a trajectory is bound to the measured joint state
        (or verified controller reference during a mid-motion replacement).
        """
        self._require_measured(contract.resources)
        queued = after is not None and after.state not in TERMINAL_GOAL_STATES
        if queued:
            reading = self._expected_reading(contract.resources[0], after)
        else:
            reading = Reading(
                self._feedback.samples[contract.resources[0]].values,
                self._feedback.execution_time_s,
                reference=reference,
            )
        if reading is not None:
            action = PROFILES[contract.action.profile].activate(contract.action, contract.limits, reading)
        else:
            action = contract.action
        conditions = contract.start_conditions
        if conditions is None:
            return action
        if conditions.required_mode is not None and conditions.required_mode != self._facts.mode:
            raise RobotError(
                "start_condition_failed",
                f"the robot is in mode {self._facts.mode!r}, not {conditions.required_mode!r}",
                mode=self._facts.mode,
            )
        source = None
        if conditions.observation_id is not None:
            issued = self._history.find(conditions.observation_id)
            if issued is None or issued.observation.context_id != self.context_id:
                raise RobotError(
                    "start_condition_failed",
                    "the source observation is not retained in the current context",
                    observation_id=conditions.observation_id,
                )
            source = issued.observation
        for rid in contract.resources:
            sample = self._feedback.samples[rid]
            if conditions.max_measurement_age_s is not None:
                # Paused stepped physics does not age a measurement; wall time counts only in realtime.
                age = self._feedback.execution_time_s - int(sample.capture_time_ns) / 1e9
                if self._facts.clock_mode == "realtime":
                    age += now - self._feedback_time
                if age > conditions.max_measurement_age_s:
                    raise RobotError(
                        "start_condition_failed",
                        f"the measurement of {rid!r} is {age:.3f} s old; the limit is "
                        f"{conditions.max_measurement_age_s:g} s",
                        resource=rid,
                        age_s=age,
                    )
            if source is not None and conditions.max_position_change is not None:
                before = source.samples.get(rid)
                if before is None or not before.valid:
                    raise RobotError(
                        "start_condition_failed",
                        f"the source observation has no valid measurement of {rid!r} to bound how far it moved",
                        resource=rid,
                    )
                predicted = self._expected_reading(rid, after) if queued else None
                if not queued or predicted is not None:
                    after_positions = (predicted.values if predicted is not None else sample.values)["positions"]
                    moved = max(
                        (abs(after_positions[name] - q) for name, q in before.values["positions"].items()), default=0.0
                    )
                    if moved > conditions.max_position_change:
                        raise RobotError(
                            "start_condition_failed",
                            f"{rid!r} moved {moved:.4f} since the source observation; the limit is "
                            f"{conditions.max_position_change:g}",
                            resource=rid,
                            moved=moved,
                            limit=conditions.max_position_change,
                        )
        return action

    def _activate(self, record: _GoalRecord, now: float) -> None:
        """Wait for any predecessor, recheck start conditions, then dispatch. A rejection causes no movement."""
        goal = record.goal
        contract = goal.contract
        if now >= record.wall_deadline:
            ending = _Ending(
                "failed",
                "precondition_changed",
                "the wall deadline passed before dispatch",
                {"code": "start_condition_failed"},
            )
            self._end_goal(record, ending, now)
            return

        replacing: _GoalRecord | None = None
        if contract.after_goal_id is not None:
            predecessor = self._goals.get(contract.after_goal_id)
            if predecessor is not None:
                pred_state = predecessor.goal.state
                if pred_state in ("accepted", "executing", "finishing"):
                    return
                if pred_state != "succeeded":
                    self._end_goal(
                        record,
                        _Ending(
                            "cancelled",
                            "predecessor_failed",
                            f"predecessor goal {contract.after_goal_id} ended {pred_state}",
                            {
                                "predecessor_goal_id": contract.after_goal_id,
                                "predecessor_state": pred_state,
                                "predecessor_reason": self._goal_reason(predecessor.goal, predecessor),
                            },
                        ),
                        now,
                    )
                    return
        elif contract.replace_goal_id is not None:
            found = self._find_goal(contract.replace_goal_id)
            if found is not None:
                target_goal, target_rec = found
                already_superseded = (
                    target_goal.state == "cancelled"
                    and target_goal.result is not None
                    and target_goal.result.reason == "superseded"
                    and target_goal.result.measurements.get("superseded_by") == goal.goal_id
                )
                if already_superseded or target_goal.state == "succeeded":
                    pass
                elif target_goal.state in ("executing", "finishing") and target_rec is not None:
                    replacing = target_rec
                else:
                    self._end_goal(
                        record,
                        _Ending(
                            "cancelled",
                            "predecessor_failed",
                            f"target goal {contract.replace_goal_id} ended {target_goal.state}",
                            {
                                "predecessor_goal_id": contract.replace_goal_id,
                                "predecessor_state": target_goal.state,
                                "predecessor_reason": self._goal_reason(target_goal, target_rec),
                            },
                        ),
                        now,
                    )
                    return

        if replacing is None:
            for rid in contract.resources:
                if self._resources[rid].status.activity not in (None, goal.goal_id):
                    return
        if record.execution_deadline is None:
            record.execution_deadline = self._feedback.execution_time_s + contract.max_execution_s
        reference = self._feedback.references.get(contract.resources[0]) if replacing is not None else None
        try:
            if self._feedback.execution_time_s >= record.execution_deadline:
                raise RobotError("start_condition_failed", "the execution deadline passed before dispatch")
            if replacing is not None:
                for rid in contract.resources:
                    status = self._resources[rid].status
                    if status.condition != "active" or status.activity != replacing.goal.goal_id:
                        raise RobotError(
                            "start_condition_failed",
                            f"resource {rid!r} is no longer executing {replacing.goal.goal_id}",
                            resource=rid,
                        )
            else:
                for rid in contract.resources:
                    condition = self._resources[rid].status.condition
                    if condition != "ready":
                        raise RobotError("start_condition_failed", f"resource {rid!r} became {condition}", resource=rid)
            action = self._check_start(contract, now, reference=reference)
        except Exception as error:
            if not isinstance(error, RobotError):
                log.exception("could not check the start conditions of goal %s", goal.goal_id)
                error = RobotError("backend_error", str(error))
            ending = _Ending("failed", "precondition_changed", error.message, {"code": error.code, **error.details})
            self._end_goal(record, ending, now)
            return
        if replacing is None:
            for rid in contract.resources:
                self._resources[rid].status.activity = goal.goal_id

        write = record.dispatch_write
        if write is None:
            # The durable marker records intent while any replaced goal keeps moving undisturbed.
            document = goal.model_dump(mode="json")
            document["state"] = "executing"
            record.dispatch_write = self._journal.record_later(self.robot_id, goal.goal_id, document, finished=False)
            return
        if not write.done():
            return
        if (error := write.exception()) is not None:
            self._end_goal(record, _Ending("failed", "journal_error", str(error)), now)
            return

        contract.action = action
        try:
            judge = self._create_judge(contract, reference=reference)
        except (TypeError, ValueError, KeyError) as error:
            self._end_goal(record, _Ending("failed", "feedback_lost", str(error)), now)
            return
        if replacing is not None:
            self._finish_goal(
                replacing,
                "cancelled",
                "superseded",
                now,
                detail=f"superseded by {goal.goal_id}",
                measurements={"superseded_by": goal.goal_id},
                stop_confirmed=None,
            )
            self._cancel_dependents(
                replacing.goal.goal_id, "cancelled", "superseded", now, exclude=frozenset({goal.goal_id})
            )
        started = self._feedback.execution_time_s
        record.evaluation = _GoalEvaluation(judge, started, started)
        command = contract
        if self._backend.reports_goal_samples:
            command = GoalCommand.model_construct(**dict(contract), judge=judge)
        goal.activated_time_ns = _ns(now)
        goal.state = "executing"
        for rid in contract.resources:
            status = self._resources[rid].status
            status.condition, status.activity, status.retained = "active", goal.goal_id, None
        try:
            self._backend.apply(goal.goal_id, command)
        except Exception as error:
            log.exception("delivery of goal %s is uncertain", goal.goal_id)
            self._end_goal(record, _Ending("failed", "backend_error", str(error)), now)
            self._dispatch_stops()
            return
        self._history.emit("goal.executing", now, goal_id=goal.goal_id)

    def _create_judge(self, contract: Contract, *, reference: JointReference | None = None) -> MotionJudge:
        reading = Reading(
            self._feedback.samples[contract.resources[0]].values,
            self._feedback.execution_time_s,
            reference=reference,
        )
        return PROFILES[contract.action.profile].judge(contract.action, contract.limits, reading)

    def _evaluate_goal(self, record: _GoalRecord) -> ActivityStatus | None:
        evaluation = record.evaluation
        judge = evaluation.judge
        if self._backend.reports_goal_samples:
            return judge.status
        contract = record.goal.contract
        rid = contract.resources[0]
        sample = self._feedback.samples[rid]
        time_s = int(sample.capture_time_ns) / 1e9
        if time_s <= evaluation.last_sample_s:
            return judge.status
        reading = Reading(
            sample.values,
            time_s,
            elapsed_s=time_s - evaluation.started_s,
            dt_s=time_s - evaluation.last_sample_s,
            still_for_s=contract.limits.settle_time_s if self._feedback.stopped.get(rid, False) else 0.0,
        )
        evaluation.last_sample_s = time_s
        return PROFILES[contract.action.profile].measure(judge, reading)

    def _progress_goals(self, now: float) -> None:
        for record in list(self._goals.values()):
            goal = record.goal
            resources = [self._resources[rid].status for rid in goal.contract.resources]
            if goal.state == "stopping":
                if any(status.condition == "stopping" for status in resources):
                    continue
                ending = record.ending
                if any(status.condition == "faulted" for status in resources):
                    self._finish_goal(
                        record,
                        "failed",
                        "stop_unconfirmed",
                        now,
                        detail=f"stopping after {ending.reason!r} was not confirmed by its deadline",
                        measurements={**ending.measurements, "cause": ending.reason},
                        stop_confirmed=False,
                    )
                else:
                    self._finish_goal(
                        record,
                        ending.state,
                        ending.reason,
                        now,
                        detail=ending.detail,
                        measurements=ending.measurements,
                        stop_confirmed=True,
                    )
            elif goal.state in ("executing", "finishing"):
                status = self._feedback.activities.get(goal.goal_id)
                if any(not self._valid(rid) for rid in goal.contract.resources):
                    self._end_goal(record, _Ending("failed", "feedback_lost", "measurements became invalid"), now)
                elif status is None:
                    self._end_goal(record, _Ending("failed", "backend_error", "the backend lost this goal"), now)
                elif status.phase == "failed":
                    ending = _Ending("failed", status.error or "backend_error", status.detail, status.measurements)
                    self._end_goal(record, ending, now)
                # The deadlines bound the goal, so they are tested before completion: a goal that is found
                # complete only after a deadline (a stalled control loop, say) did not finish within it.
                elif self._feedback.execution_time_s >= record.execution_deadline:
                    self._end_goal(record, _Ending("failed", "execution_deadline_exceeded"), now)
                elif now >= record.wall_deadline:
                    self._end_goal(record, _Ending("failed", "wall_deadline_exceeded"), now)
                else:
                    try:
                        verdict = self._evaluate_goal(record)
                    except (KeyError, TypeError, ValueError) as error:
                        self._end_goal(record, _Ending("failed", "feedback_lost", str(error)), now)
                        continue
                    if verdict is not None and verdict.phase == "failed":
                        self._end_goal(
                            record, _Ending("failed", verdict.error, verdict.detail, verdict.measurements), now
                        )
                        continue
                    if status.phase != "done":
                        goal.state = status.phase
                        continue
                    if verdict is None or verdict.phase != "done":
                        ending = _Ending("failed", "completion_unverified", "measurements did not verify completion")
                        self._end_goal(record, ending, now)
                        continue
                    for resource in resources:
                        resource.condition, resource.activity = "ready", None
                    self._set_holding(
                        (self._resources[rid] for rid in goal.contract.resources),
                        kind="hold",
                        source=goal.goal_id,
                        profile=goal.contract.action.profile,
                    )
                    self._finish_goal(
                        record,
                        "succeeded",
                        "completed",
                        now,
                        measurements={**status.measurements, **verdict.measurements},
                        observation_id=self._issue(now).observation_id,
                    )

    def _end_goal(self, record: _GoalRecord, ending: _Ending, now: float, *, cascade: bool = True) -> None:
        goal = record.goal
        if goal.state == "accepted":
            # Never dispatched: nothing moved, so there is nothing to stop.
            for rid in goal.contract.resources:
                if self._resources[rid].status.activity == goal.goal_id:
                    self._resources[rid].status.activity = None
            self._finish_goal(
                record, ending.state, ending.reason, now, detail=ending.detail, measurements=ending.measurements
            )
        elif goal.state in ("executing", "finishing"):
            record.ending = ending
            goal.state = "stopping"
            self._begin_stop(goal.contract.resources, now)
            self._history.emit("goal.stopping", now, goal_id=goal.goal_id, reason=ending.reason)
        else:
            return
        if cascade and ending.state != "succeeded":
            self._cancel_dependents(goal.goal_id, ending.state, ending.reason, now)

    def _cancel_dependents(
        self,
        failed_id: str,
        state: str,
        reason: str,
        now: float,
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> None:
        """Cancel every accepted goal whose causal chain depends on a goal that did not succeed."""
        failed: dict[str, tuple[str, str]] = {failed_id: (state, reason)}
        for record in self._goals.values():
            goal = record.goal
            if goal.state != "accepted" or goal.goal_id in exclude:
                continue
            parent_id = goal.contract.after_goal_id or goal.contract.replace_goal_id
            if parent_id not in failed:
                continue
            parent_state, parent_reason = failed[parent_id]
            for rid in goal.contract.resources:
                if self._resources[rid].status.activity == goal.goal_id:
                    self._resources[rid].status.activity = None
            self._finish_goal(
                record,
                "cancelled",
                "predecessor_failed",
                now,
                detail=f"predecessor goal {parent_id} ended {parent_state} ({parent_reason})",
                measurements={
                    "predecessor_goal_id": parent_id,
                    "predecessor_state": parent_state,
                    "predecessor_reason": parent_reason,
                },
            )
            failed[goal.goal_id] = ("cancelled", "predecessor_failed")

    def _finish_goal(
        self,
        record: _GoalRecord,
        state: str,
        reason: str,
        now: float,
        *,
        detail: str | None = None,
        measurements: dict[str, Any] | None = None,
        observation_id: str | None = None,
        stop_confirmed: bool | None = None,
    ) -> None:
        goal = record.goal
        goal.state = state
        goal.finished_time_ns = _ns(now)
        goal.result = GoalResult(
            reason=reason,
            detail=detail,
            measurements=measurements or {},
            completion_observation_id=observation_id,
            stop_confirmed=stop_confirmed,
        )
        retained = self._resources[goal.contract.resources[0]].status.retained
        goal.retained = retained.model_copy() if retained else None
        self._journal.record_later(self.robot_id, goal.goal_id, goal.model_dump(mode="json"), finished=True)
        self._history.emit(f"goal.{state}", now, goal_id=goal.goal_id, reason=reason)

    # ------------------------------------------------------------------
    # Target streams: open -> stopping -> closed. They never report success.
    # ------------------------------------------------------------------

    def open_stream(self, principal: Principal, request: StreamOpen) -> Stream:
        body = request.model_dump(mode="json")
        with self._admission:
            prior = self._replay(principal, request.request_id, body, "stream")
            if prior is not None:
                return self.stream(prior.operation_id)
            with self._lock:
                now = self._clock.now()
                self._expire(now)
                self._authority(principal, request.session_id, request.context_id)
                binding = find_binding(self._facts.bindings, request.profile, request.target, "stream")
                self._require_available(binding)
                limits = resolve_stream(request, binding)
                self._check_free(binding.resources)
                self._require_measured(binding.resources)
                # The initial target is sequence zero and obeys the same freshness rule.
                expires_at, basis_sequence = self._basis_expiry(
                    request.basis_observation_id, binding.resources, request.update_validity_wall_s, now, 0
                )
                stream_id = _new_id("str")
                stream = Stream(
                    stream_id=stream_id,
                    request_id=request.request_id,
                    robot_id=self.robot_id,
                    context_id=self.context_id,
                    session_id=request.session_id,
                    principal=principal.name,
                    profile=request.profile,
                    target=request.target,
                    resources=binding.resources,
                    limits=limits,
                    update_validity_wall_s=request.update_validity_wall_s,
                    state="open",
                    accepted_sequence=0,
                    expires_at_ns=_ns(expires_at),
                    expires_in_s=expires_at - now,
                    update_path=f"/v1/robots/{self.robot_id}/streams/{stream_id}/target",
                    persistent_channel=NO_CHANNEL,
                )
                stop_generation = self._stop_generation
            self._admit(principal, "stream", stream_id, request_id=request.request_id, request=body, document=stream)
            with self._lock:
                now = self._clock.now()
                self._expire(now)
                self._streams[stream_id] = _StreamRecord(
                    stream,
                    expires_at,
                    basis_sequence,
                    mailbox=request.initial_target,
                    acks=OrderedDict(),
                )
                record = self._streams[stream_id]
                if reason := self._admission_interrupted(stream.context_id, stop_generation, stream.session_id):
                    self._close_stream(record, "interrupted", now, reason)
                    return self._stream_view(record, now)
                if now >= expires_at:
                    self._close_stream(record, "expired", now, "target expired during admission")
                    return self._stream_view(record, now)
                for rid in binding.resources:
                    self._resources[rid].status.activity = stream_id
                self._history.emit("stream.opened", now, stream_id=stream_id, profile=request.profile)
                return self._stream_view(record, now)

    def update_stream(self, principal: Principal, stream_id: str, request: StreamUpdate) -> StreamUpdateResult:
        with self._lock:
            now = self._clock.now()
            # Expiry runs first, under the same lock: a late packet cannot revive an expired stream.
            self._expire(now)
            record = self._streams.get(stream_id)
            if record is not None:
                stream = record.stream
                # A stream stays bound to the session and context that opened it.
                self._authority(principal, request.session_id, request.context_id)
                if request.session_id != stream.session_id:
                    raise RobotError("permission_denied", "this stream belongs to another session")
                body_digest = digest(request.model_dump(mode="json"))
                known = record.acks.get(request.sequence)
                if known is not None:
                    if known[0] != body_digest:
                        raise RobotError(
                            "sequence_conflict",
                            f"sequence {request.sequence} was already accepted with different data",
                            sequence=request.sequence,
                        )
                    # The original acknowledgement; expiry is not extended.
                    return StreamUpdateResult(ack=known[1], duplicate=True, stream=self._stream_view(record, now))
                if stream.state != "open":
                    raise RobotError(
                        "stream_closed",
                        f"this stream is {stream.state}; a late update cannot reopen it",
                        state=stream.state,
                        closure=stream.closure or record.ending,
                    )
                if request.sequence <= stream.accepted_sequence:
                    raise RobotError(
                        "sequence_rejected",
                        f"sequence {request.sequence} is not above the accepted sequence {stream.accepted_sequence}",
                        accepted_sequence=stream.accepted_sequence,
                    )
                check_velocity(request.target, stream.limits)
                expires_at, basis_sequence = self._basis_expiry(
                    request.basis_observation_id,
                    stream.resources,
                    stream.update_validity_wall_s,
                    now,
                    record.basis_sequence,
                )
                ack = TargetAck(
                    sequence=request.sequence,
                    basis_observation_id=request.basis_observation_id,
                    expires_at_ns=_ns(expires_at),
                )
                record.acks[request.sequence] = (body_digest, ack)
                while len(record.acks) > 16:
                    record.acks.popitem(last=False)
                # Superseded targets are not replayed: only the newest waits for the controller tick.
                record.mailbox = request.target
                record.expires_at = expires_at
                record.basis_sequence = basis_sequence
                stream.accepted_sequence = request.sequence
                return StreamUpdateResult(ack=ack, duplicate=False, stream=self._stream_view(record, now))
        stored = Stream.model_validate(self._stored(stream_id, "stream"))
        raise RobotError("stream_closed", "this stream is closed", state=stored.state, closure=stored.closure)

    def stream(self, stream_id: str) -> Stream:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            record = self._streams.get(stream_id)
            if record is not None:
                return self._stream_view(record, now)
        return Stream.model_validate(self._stored(stream_id, "stream"))

    def close_stream(self, principal: Principal, stream_id: str, authority: Authority) -> Stream:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            record = self._streams.get(stream_id)
            if record is not None:
                if record.stream.state == "open":
                    if not self._may_end(principal, authority) and record.stream.session_id != authority.session_id:
                        raise RobotError("permission_denied", "this stream belongs to another session")
                    self._end_stream(record, "client_closed", now)
                return self._stream_view(record, now)
        return Stream.model_validate(self._stored(stream_id, "stream"))

    @staticmethod
    def _stream_view(record: _StreamRecord, now: float) -> Stream:
        remaining = max(0.0, record.expires_at - now) if record.stream.state == "open" else 0.0
        return record.stream.model_copy(
            update={"expires_at_ns": _ns(record.expires_at), "expires_in_s": remaining}, deep=True
        )

    def _basis_expiry(
        self, observation_id: str, resources: Sequence[str], validity_s: float, now: float, minimum_sequence: int
    ) -> tuple[float, int]:
        """When a target based on this observation expires, or why the basis is unusable.

        Expiry is the observation's issue time plus the validity window. Network
        delay consumes that window; receipt does not restart it.
        """
        issued = self._history.find(observation_id)
        if issued is None or issued.observation.context_id != self.context_id:
            raise RobotError(
                "basis_invalid",
                "the basis observation was not issued by this runtime in the current context, or is no longer retained",
                observation_id=observation_id,
            )
        if issued.observation.sequence < minimum_sequence:
            raise RobotError(
                "basis_invalid",
                "the basis observation is older than the previous accepted basis",
                sequence=issued.observation.sequence,
                previous_sequence=minimum_sequence,
            )
        for rid in resources:
            sample = issued.observation.samples.get(rid)
            if sample is None or not sample.valid:
                raise RobotError("basis_invalid", f"the basis observation has no valid measurement of {rid!r}")
        # Capture freshness is separate from issue time: an observation issued from old readings is not a basis.
        capture_age = issued.issued_at - issued.measured_at
        if self._facts.clock_mode == "realtime" and capture_age > validity_s:
            raise RobotError(
                "basis_invalid",
                f"the basis observation's measurements were already {capture_age:.3f} s old when it was issued",
                capture_age_s=capture_age,
            )
        expires_at = issued.issued_at + validity_s
        if now >= expires_at:
            raise RobotError(
                "target_expired",
                f"the basis observation was issued {now - issued.issued_at:.3f} s ago; targets stay valid for "
                f"{validity_s:g} s",
                age_s=now - issued.issued_at,
                update_validity_wall_s=validity_s,
            )
        return expires_at, issued.observation.sequence

    def _apply_targets(self, now: float) -> None:
        """Hand each open stream's newest accepted target to its controller."""
        for record in list(self._streams.values()):
            stream = record.stream
            if stream.state != "open" or record.mailbox is None:
                continue
            # Authority and expiry were rechecked at the top of this tick; the measurements are rechecked here.
            # A stream that has not moved anything yet closes without a command ever reaching the controller.
            if any(not self._valid(rid) for rid in stream.resources):
                self._end_stream(record, "failed", now, "feedback_lost")
                continue
            first = stream.applied_sequence is None
            if first:
                write = record.dispatch_write
                if write is None:
                    document = self._stream_view(record, now).model_dump(mode="json")
                    document["dispatch_pending"] = True
                    record.dispatch_write = self._journal.record_later(
                        self.robot_id, stream.stream_id, document, finished=False
                    )
                    continue
                if not write.done():
                    continue
                if (error := write.exception()) is not None:
                    self._end_stream(record, "failed", now, f"journal_error: {error}")
                    continue
                for rid in stream.resources:
                    status = self._resources[rid].status
                    status.condition, status.retained = "active", None
            try:
                self._backend.track(stream.stream_id, stream.target, record.mailbox, stream.limits)
            except Exception as error:
                log.exception("delivery of stream %s is uncertain", stream.stream_id)
                self._stop_stream(record, "failed", now, f"backend_error: {error}")
                self._dispatch_stops()
                continue
            stream.applied_sequence, record.mailbox = stream.accepted_sequence, None
            if first:
                document = self._stream_view(record, now).model_dump(mode="json")
                document["dispatch_pending"] = False
                self._journal.record_later(self.robot_id, stream.stream_id, document, finished=False)

    def _progress_streams(self, now: float) -> None:
        for record in list(self._streams.values()):
            stream = record.stream
            if stream.state == "stopping":
                conditions = [self._resources[rid].status.condition for rid in stream.resources]
                if "stopping" in conditions:
                    continue
                if "faulted" in conditions:
                    self._close_stream(record, "failed", now, "stop_unconfirmed")
                else:
                    self._close_stream(record, record.ending, now)
            elif stream.state == "open" and stream.applied_sequence is not None:
                status = self._feedback.activities.get(stream.stream_id)
                if any(not self._valid(rid) for rid in stream.resources):
                    self._end_stream(record, "failed", now, "feedback_lost")
                elif status is None:
                    self._end_stream(record, "failed", now, "backend_error")
                elif status.phase == "failed":
                    self._end_stream(record, "failed", now, status.error or "backend_error")

    def _end_stream(self, record: _StreamRecord, closure: str, now: float, detail: str | None = None) -> None:
        """Closure, expiry, failure, and session loss all come here, and all invoke stopping."""
        stream = record.stream
        if stream.state != "open":
            return
        record.mailbox = None  # an unapplied target is never dispatched
        stream.detail = detail
        if stream.applied_sequence is None:
            for rid in stream.resources:
                self._resources[rid].status.activity = None
            self._close_stream(record, closure, now, detail)
        else:
            self._stop_stream(record, closure, now, detail)

    def _stop_stream(self, record: _StreamRecord, closure: str, now: float, detail: str | None) -> None:
        stream = record.stream
        record.mailbox = None
        record.ending = closure
        stream.detail, stream.state = detail, "stopping"
        self._begin_stop(stream.resources, now)
        self._history.emit("stream.stopping", now, stream_id=stream.stream_id, closure=closure)

    def _close_stream(self, record: _StreamRecord, closure: str, now: float, detail: str | None = None) -> None:
        stream = record.stream
        stream.state, stream.closure = "closed", closure
        stream.detail = detail or stream.detail
        record.mailbox = None
        view = self._stream_view(record, now)
        self._journal.record_later(self.robot_id, stream.stream_id, view.model_dump(mode="json"), finished=True)
        self._history.emit("stream.closed", now, stream_id=stream.stream_id, closure=closure)

    # ------------------------------------------------------------------
    # Resources: availability, stopping, and what stays held
    # ------------------------------------------------------------------

    def stop(self, principal: Principal, authority: Authority) -> Operation:
        """Stop current commands and retained targets under the mode's fallback.

        An owner stops with its session. An operator needs neither a session nor
        a matching context, so restart or expiry cannot disable stopping.
        """
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            self._may_end(principal, authority)
            self._stop_generation += 1
            operation = self._new_operation("stop", principal, now)
            self._operations[operation.operation_id] = operation
            # Every resource is awaited, including a faulted resource whose standstill could not be confirmed.
            self._awaited[operation.operation_id] = list(self._resources)
            self._begin_stop(list(self._resources), now)
            for record in self._goals.values():
                if record.goal.state not in TERMINAL_GOAL_STATES:
                    self._end_goal(record, _Ending("cancelled", "stopped"), now, cascade=False)
            for stream in self._streams.values():
                if stream.stream.state == "open":
                    self._end_stream(stream, "interrupted", now, "stopped")
            self._history.emit("stop.requested", now, operation_id=operation.operation_id, principal=principal.name)
            return operation.model_copy(deep=True)

    def _availability(self, binding: ProfileBinding) -> ProfileBinding:
        """A supported profile is not always available now; say why."""
        if not binding.available:
            return binding
        for rid in binding.resources:
            status = self._resources[rid].status
            if status.condition in ("faulted", "recovering"):
                reason = f"resource {rid!r} is {status.condition} ({status.fault}); an operator must complete recovery"
                return binding.model_copy(update={"available": False, "unavailable_reason": reason})
        return binding

    def _require_available(self, binding: ProfileBinding) -> None:
        current = self._availability(binding)
        if not current.available:
            code = "profile_unavailable" if not binding.available else "resource_faulted"
            raise RobotError(
                code,
                f"{binding.profile} on {binding.target!r} is unavailable: {current.unavailable_reason}",
                profile=binding.profile,
                target=binding.target,
            )

    def _check_free(
        self,
        resources: Sequence[str],
        *,
        after: Goal | None = None,
        ignore: Iterable[str] = (),
    ) -> None:
        """One active goal or stream per resource group, or an explicit causal chain / replacement."""
        if self._transition_in_progress is not None:
            operation = self._transition_in_progress.operation
            raise RobotError("resource_busy", f"{operation.kind} is still running", operation_id=operation.operation_id)
        chain = self._ancestor_chain(after, follow_replace=True)
        exempt = {goal.goal_id for goal in chain} | set(ignore)
        for anc in chain:
            if anc.state == "accepted" and anc.contract.replace_goal_id is not None:
                exempt |= self._descendants(anc.contract.replace_goal_id, include_replace=False)
        for rid in resources:
            status = self._resources[rid].status
            if status.condition == "stopping" or (status.activity is not None and status.activity not in exempt):
                what = "stopping" if status.condition == "stopping" else f"in use by {status.activity}"
                raise RobotError("resource_busy", f"resource {rid!r} is {what}", resource=rid, activity=status.activity)
        for other in self._resources.values():
            busy = other.status
            if busy.id in resources:
                continue
            busy_now = busy.condition == "stopping" or (busy.activity is not None and busy.activity not in exempt)
            if not busy_now:
                continue
            for rid in resources:
                if not any(rid in group and busy.id in group for group in self._facts.concurrent):
                    raise RobotError(
                        "resource_busy",
                        f"{rid!r} cannot run while {busy.id!r} is active: that controller combination is not validated",
                        resource=rid,
                        active=busy.id,
                    )
        for record in self._goals.values():
            other_goal = record.goal
            if other_goal.state in TERMINAL_GOAL_STATES or other_goal.goal_id in exempt:
                continue
            for rid in resources:
                if rid in other_goal.contract.resources:
                    raise RobotError(
                        "resource_busy",
                        f"resource {rid!r} already has queued goal {other_goal.goal_id}",
                        resource=rid,
                        activity=other_goal.goal_id,
                    )
                for other_rid in other_goal.contract.resources:
                    if not any(rid in group and other_rid in group for group in self._facts.concurrent):
                        raise RobotError(
                            "resource_busy",
                            f"{rid!r} cannot run concurrently with queued goal {other_goal.goal_id} on {other_rid!r}",
                            resource=rid,
                            active=other_rid,
                            activity=other_goal.goal_id,
                        )

    def _begin_stop(self, resources: Sequence[str], now: float) -> None:
        for rid in resources:
            resource = self._resources[rid]
            if resource.status.condition == "stopping":
                # Its stop is already queued or commanded. Commanding it again would restart the
                # controller's standstill measurement, and repeated stops could then never be confirmed.
                continue
            if resource.status.condition in ("ready", "active"):
                resource.status.condition = "stopping"
                resource.deadline = now + resource.stop_deadline_wall_s
                resource.commanded_at = None
            # A faulted resource stays faulted, but the fallback is still commanded.
            self._stop_queue.add(rid)

    def _dispatch_stops(self) -> None:
        if not self._stop_queue:
            return
        try:
            self._backend.stop(sorted(self._stop_queue))
        except Exception:
            # Not delivered, so nothing can confirm it. Resources that are stopping stay queued and the
            # stop is sent again next tick, until their deadline faults them.
            log.exception("backend stop failed; it will be sent again until the stopping deadline")
            self._stop_queue = {rid for rid in self._stop_queue if self._resources[rid].status.condition == "stopping"}
            return
        for rid in self._stop_queue:
            self._resources[rid].commanded_at = self._state_sequence
        self._stop_queue.clear()

    def _settle_resources(self, now: float) -> None:
        """Move stopping or recovering resources to ready on measured standstill, or to faulted."""
        for resource in self._resources.values():
            status = resource.status
            if status.condition not in ("stopping", "recovering"):
                continue
            measured = self._valid(status.id) and self._feedback.stopped.get(status.id, False)
            # Standstill counts only in a reading taken after the command it confirms.
            commanded = resource.commanded_at is not None and self._state_sequence > resource.commanded_at
            if measured and commanded:
                status.condition, status.activity, status.fault = "ready", None, None
                resource.deadline = resource.commanded_at = None
                self._set_holding([resource])
                self._history.emit("resource.ready", now, resource=status.id)
            elif now >= resource.deadline:
                # Unknown motion is reported as unknown, not as a stationary pose.
                status.fault = "stop_unconfirmed" if status.condition == "stopping" else "recovery_unconfirmed"
                status.condition, status.activity, status.retained = "faulted", None, None
                resource.deadline = resource.commanded_at = None
                self._history.emit("resource.faulted", now, resource=status.id, fault=status.fault)

    def _set_holding(
        self,
        resources: Iterable[_Resource],
        *,
        kind: Literal["hold", "fallback"] = "fallback",
        source: str | None = None,
        profile: str | None = None,
    ) -> None:
        """Record what each controller reports it is maintaining now that nothing commands it."""
        for resource in resources:
            target = self._feedback.holding.get(resource.status.id, {})
            resource.status.retained = Retained(kind=kind, source=source, profile=profile, target=target)

    # ------------------------------------------------------------------
    # Management: recovery and world transitions, performed on the control thread
    # ------------------------------------------------------------------

    def manage(self, principal: Principal, request: ManagementRequest) -> Operation:
        """Queue an advertised operation. The control thread performs it on its next tick."""
        if not principal.operator:
            raise RobotError("permission_denied", "management operations need operator permission")
        body = request.model_dump(mode="json")
        with self._admission:
            prior = self._replay(principal, request.request_id, body, "management")
            if prior is not None:
                return self.operation(prior.operation_id)
            with self._lock:
                now = self._clock.now()
                self._expire(now)
                self._check_context(request.context_id)
                if request.operation not in self._facts.management:
                    raise RobotError(
                        "unsupported_operation",
                        f"{request.operation} is not available on this robot",
                        operation=request.operation,
                        supported=self._facts.management,
                    )
                operation = self._new_operation(request.operation, principal, now)
                # Journalled before it is kept: a request the journal refuses leaves no operation behind.
                stop_generation = self._stop_generation
            self._admit(
                principal,
                "management",
                operation.operation_id,
                request_id=request.request_id,
                request=body,
                document=operation,
            )
            with self._lock:
                self._operations[operation.operation_id] = operation
                now = self._clock.now()
                if reason := self._admission_interrupted(operation.context_id, stop_generation):
                    self._finish_operation(operation, now, reason, "interrupted while saving the request")
                    return operation.model_copy(deep=True)
                self._management.append((operation, request))
                self._history.emit(
                    "management.requested", now, operation_id=operation.operation_id, operation=request.operation
                )
                return operation.model_copy(deep=True)

    def operation(self, operation_id: str) -> Operation:
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is not None:
                return operation.model_copy(deep=True)
        return Operation.model_validate(self._stored(operation_id, "management"))

    def _new_operation(self, kind: str, principal: Principal, now: float) -> Operation:
        return Operation(
            operation_id=_new_id("op"),
            kind=kind,
            robot_id=self.robot_id,
            principal=principal.name,
            state="running",
            context_id=self.context_id,
            requested_time_ns=_ns(now),
        )

    def _run_management(self, now: float) -> None:
        pending = self._transition_in_progress
        if pending is not None:
            operation, result = pending.operation, pending.result
            if pending.deadline is not None and now >= pending.deadline and not result.done():
                if not result.cancel():
                    operation.detail = "waiting for confirmation of the command sent to the backend"
                pending.deadline = None
            if not result.done():
                return
            self._transition_in_progress = None
            if result.cancelled():
                self._finish_operation(operation, now, "backend_error", "the management command timed out in the queue")
            else:
                try:
                    self._complete_transition(operation, now, result.result(), reset_resources=False)
                except RobotError as error:
                    self._finish_operation(operation, now, error.code, error.message)
                except Exception as error:
                    log.exception("management operation %s failed", operation.kind)
                    self._finish_operation(operation, now, "backend_error", str(error))

        while self._management and self._transition_in_progress is None:
            operation, request = self._management.popleft()
            try:
                # An operation queued behind one that changed the context was not addressed to the new one.
                self._check_context(operation.context_id)
                if operation.kind == "recover":
                    self._start_recovery(operation, request, now)
                else:
                    self._transition(operation, request, now)
            except RobotError as error:
                self._finish_operation(operation, now, error.code, error.message)
            except Exception as error:
                log.exception("management operation %s failed", operation.kind)
                self._finish_operation(operation, now, "backend_error", str(error))

    def _start_recovery(self, operation: Operation, request: ManagementRequest, now: float) -> None:
        faulted = [rid for rid, r in self._resources.items() if r.status.condition == "faulted"]
        if not faulted:
            self._finish_operation(operation, now, None, None, {"recovered": []})
            return
        self._backend.manage("recover", {**request.parameters, "resources": faulted})
        # Clearing a flag is not recovery: each resource must again show valid, measured standstill.
        for rid in faulted:
            resource = self._resources[rid]
            resource.status.condition = "recovering"
            resource.deadline = now + resource.stop_deadline_wall_s
            resource.commanded_at = self._state_sequence
        self._awaited[operation.operation_id] = faulted
        operation.outcome = {"recovering": faulted}

    def _transition(self, operation: Operation, request: ManagementRequest, now: float) -> None:
        """A barrier such as world reset: end everything, change the world, start a new context."""
        for record in self._goals.values():
            if record.goal.state not in TERMINAL_GOAL_STATES:
                self._finish_goal(
                    record, "interrupted", operation.kind, now, detail="interrupted by a management operation"
                )
        for stream in self._streams.values():
            if stream.stream.state != "closed":
                self._close_stream(stream, "interrupted", now, operation.kind)
        for operation_id in list(self._awaited):
            self._finish_operation(self._operations[operation_id], now, "interrupted", f"overtaken by {operation.kind}")
        self._awaited.clear()
        # Queue the fallback first: if the backend refuses the transition, this tick still stops old motion.
        self._begin_stop(list(self._resources), now)
        outcome = self._backend.manage(operation.kind, request.parameters)
        if isinstance(outcome, Future):
            self._transition_in_progress = _Management(operation, outcome, now + 60.0)
        else:
            self._complete_transition(operation, now, outcome, reset_resources=True)

    def _complete_transition(
        self, operation: Operation, now: float, outcome: dict[str, Any], *, reset_resources: bool
    ) -> None:
        self._facts = self._backend.describe()
        if reset_resources:
            self._stop_queue.clear()
            self._feedback = self._read_backend()
            self._feedback_time = now
            self._state_sequence += 1
            for resource in self._resources.values():
                status, measured = resource.status, self._valid(resource.status.id)
                status.condition, status.activity = ("ready" if measured else "faulted"), None
                status.fault, status.retained = (None if measured else "feedback_invalid"), None
                resource.deadline = resource.commanded_at = None
            self._set_holding(r for r in self._resources.values() if r.status.condition == "ready")
        self._change_context(operation.kind, now)
        operation.new_context_id = self.context_id
        self._finish_operation(operation, now, None, None, outcome)

    def _progress_operations(self, now: float) -> None:
        """Finish stop and recovery operations once every resource they wait on has settled."""
        for operation_id, awaited in list(self._awaited.items()):
            conditions = {rid: self._resources[rid].status.condition for rid in awaited}
            if any(c in ("stopping", "recovering") for c in conditions.values()):
                continue
            del self._awaited[operation_id]
            operation = self._operations[operation_id]
            unconfirmed = sorted(rid for rid, c in conditions.items() if c == "faulted")
            reason = None
            if unconfirmed:
                reason = "stop_unconfirmed" if operation.kind == "stop" else "recovery_failed"
            if operation.kind == "recover" and len(unconfirmed) < len(awaited):
                # A resource is usable again on re-established feedback and standstill, so authority starts
                # over, also when the recovery failed for other resources.
                self._change_context("recovered", now)
                operation.new_context_id = self.context_id
            detail = f"no measured standstill on {', '.join(unconfirmed)} before the deadline" if unconfirmed else None
            self._finish_operation(operation, now, reason, detail)

    def _finish_operation(
        self,
        operation: Operation,
        now: float,
        reason: str | None,
        detail: str | None,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        operation.state = "failed" if reason else "succeeded"
        operation.reason, operation.detail = reason, detail
        operation.outcome = outcome if outcome is not None else operation.outcome
        operation.resources = {rid: r.status.condition for rid, r in self._resources.items()}
        operation.finished_time_ns = _ns(now)
        if operation.kind != "stop":
            self._journal.record_later(
                self.robot_id, operation.operation_id, operation.model_dump(mode="json"), finished=True
            )
        self._history.emit(
            f"operation.{operation.state}", now, operation_id=operation.operation_id, operation=operation.kind
        )

    # ------------------------------------------------------------------
    # Journal: request identity and stored results
    # ------------------------------------------------------------------

    def _admission_interrupted(
        self, context_id: str, stop_generation: int, session_id: str | None = None
    ) -> str | None:
        """A command saved without holding the control lock still needs current authority."""
        if context_id != self.context_id:
            return "stale_context"
        if stop_generation != self._stop_generation:
            return "stopped"
        if session_id is not None and self._transition_in_progress is not None:
            return "resource_busy"
        if session_id is not None and (self._session is None or self._session.public.session_id != session_id):
            return "session_invalid"
        return None

    def _replay(self, principal: Principal, request_id: str, body: dict[str, Any], kind: str) -> Entry | None:
        """Find an earlier request with this identity. Lookup precedes every context guard."""
        entry = self._journal.lookup(principal.name, self.robot_id, request_id)
        if entry is not None and (entry.kind != kind or entry.body_sha256 != digest(body)):
            # The refusal names the operation: whoever repeats an ID wants to know what its request did.
            raise RobotError(
                "request_conflict",
                f"request_id {request_id!r} already names {entry.operation_id}, and this request differs from it",
                request_id=request_id,
                kind=entry.kind,
                operation_id=entry.operation_id,
            )
        return entry

    def _admit(
        self,
        principal: Principal,
        kind: str,
        operation_id: str,
        *,
        request_id: str,
        request: dict[str, Any],
        document: Session | Goal | Stream | Operation,
    ) -> None:
        """Journal a new, unfinished operation under its caller's request ID, in the current context."""
        self._journal.admit(
            principal=principal.name,
            robot_id=self.robot_id,
            request_id=request_id,
            kind=kind,
            operation_id=operation_id,
            context_id=document.context_id,
            request=request,
            document=document.model_dump(mode="json"),
        )

    def _stored(self, operation_id: str, kind: str) -> dict[str, Any]:
        """The journal's copy of a goal, stream, or operation this incarnation no longer holds."""
        prefix = "op" if kind == "management" else "str" if kind == "stream" else kind
        if operation_id.startswith(f"{prefix}-"):
            outcome = self._journal.outcome(self.robot_id, operation_id)
            if outcome is not None:
                return outcome
        entry = self._journal.find(self.robot_id, operation_id)
        if entry is None or entry.kind != kind:
            noun = "operation" if kind == "management" else kind
            raise RobotError("not_found", f"no {noun} with id {operation_id}")
        return entry.document

    def _trim(self, records: dict[str, Any], finished) -> None:
        """Keep memory bounded; finished records dropped here are still served from the journal."""
        surplus = len(records) - self._limits.operation_retention
        for key in [key for key, value in records.items() if finished(value)][: max(surplus, 0)]:
            del records[key]

    def _interrupt_previous_boot(self) -> None:
        """Operations an earlier incarnation left unfinished are interrupted, never replayed."""
        for entry in self._journal.unfinished(self.robot_id):
            document = entry.document
            if entry.kind == "goal":
                goal = Goal.model_validate(document)
                # The journal cannot prove whether a dispatched command reached the motors.
                dispatch = "not_dispatched" if goal.state == "accepted" else "uncertain"
                goal.state = "interrupted"
                goal.result = GoalResult(
                    reason="runtime_restart",
                    detail="the runtime restarted before this goal finished; it was not replayed",
                    measurements={"dispatch": dispatch},
                )
                document = goal.model_dump(mode="json")
            elif entry.kind == "stream":
                pending = document.pop("dispatch_pending", None)
                if pending or (pending is None and document["applied_sequence"] is not None):
                    detail = "runtime_restart: a target may have been dispatched"
                    document["applied_sequence"] = None
                elif document["applied_sequence"] is None:
                    detail = "runtime_restart: no target had been dispatched"
                else:
                    detail = "runtime_restart: targets were dispatched; later sequence numbers were not recorded"
                document = {**document, "state": "closed", "closure": "interrupted", "detail": detail}
            elif entry.kind == "session":
                document = {**document, "state": "revoked", "reason": "runtime_restart", "expires_in_s": 0.0}
            elif entry.kind == "management":
                document = {**document, "state": "failed", "reason": "runtime_restart"}
            self._journal.record(self.robot_id, entry.operation_id, document, finished=True)
