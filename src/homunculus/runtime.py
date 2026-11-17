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
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Self

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
    JointTrajectoryGoal,
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
    ActivityStatus,
    activate_trajectory,
    check_velocity,
    find_binding,
    resolve_action,
    resolve_stream,
    stated_duration_s,
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
    # Host time the backend did not turn into execution time, in total for the current world.
    lag_s: float | None = None


@dataclass
class Frame:
    """A recorded image and its bytes. The runtime stores the bytes without decoding them."""

    image: CameraImage
    data: bytes


class StartupError(Exception):
    """A robot or service cannot be served as configured. The message says why, for the person who started it."""


class Backend(ABC):
    """What the runtime needs from a robot or simulator. Subclass it to add your own.

    Implement the four abstract methods. The others are needed only for what describe() advertises: track() for
    a binding with a stream mode, capture() for a camera sensor, manage() for each management operation. The
    runtime never calls them for anything not advertised, so their defaults only report a backend that
    advertises more than it implements.

    Every method except capture() is called only from the control thread, inside tick(), and must return
    promptly; a backend that does blocking I/O owns its own worker. Samples are stamped on the clock that
    execution_time_s reads. Success is not the backend's to decide: it reports what the judges in profiles.py
    decide from its measurements.
    """

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

    def manage(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Perform an advertised management operation. May block briefly."""
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
class _GoalRecord:
    goal: Goal
    wall_deadline: float
    execution_deadline: float
    ending: _Ending | None = None


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
        self._facts = backend.describe()
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

        self._session: _Session | None = None
        self._goals: dict[str, _GoalRecord] = {}
        self._streams: dict[str, _StreamRecord] = {}
        self._operations: dict[str, Operation] = {}
        self._awaited: dict[str, list[str]] = {}  # operation id -> resources it waits on
        self._management: list[tuple[Operation, ManagementRequest]] = []

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

            self._feedback = self._read_backend()
            self._feedback_time = now
            self._state_sequence += 1

            self._expire(now)
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
            self._trim(self._goals, lambda r: r.goal.state in TERMINAL_GOAL_STATES)
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
                resources=[r.status.model_copy(deep=True) for r in self._resources.values()],
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

    def _issue(self, now: float, images: Sequence[ImageHandle] = ()) -> Observation:
        return self._history.issue(
            context_id=self.context_id,
            now=now,
            measured_at=self._feedback_time,
            state_sequence=self._state_sequence,
            samples=self._feedback.samples,
            resources=[r.status.model_copy(deep=True) for r in self._resources.values()],
            images=images,
        )

    # ------------------------------------------------------------------
    # Sessions: who may command
    # ------------------------------------------------------------------

    def open_session(self, principal: Principal, request: SessionOpen) -> Session:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            body = request.model_dump(mode="json")
            prior = self._replay(principal, request.request_id, body, "session")
            if prior is not None:
                return self._session_view(prior.operation_id, now) or Session.model_validate(prior.document)
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
            self._session = _Session(session, now + lifetime)
            self._history.emit("session.opened", now, session_id=session.session_id, principal=principal.name)
            return session.model_copy()

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
            ended = self._ended_session(session_id)
            if ended is not None and ended.principal == principal.name:
                return ended  # closing it again changes nothing
            session = self._own_session(principal, session_id)
            self._end_session("closed", "client_closed", now)
            return session.public.model_copy()

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
        entry = self._journal.find(self.robot_id, session_id or "")
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
            ended = self._ended_session(session_id)
            state = ended.state if ended else "unknown"
            raise RobotError(
                "session_invalid",
                f"session {session_id} is {state}; open a new session to command this robot",
                session_id=session_id,
                state=state,
                reason=ended.reason if ended else None,
            )
        if session.public.principal != principal.name:
            raise RobotError("permission_denied", "this session belongs to another principal")
        return session

    def _end_session(self, state: str, reason: str, now: float) -> None:
        """Revoke authority and stop what it commanded. Retained holds stay in place."""
        session = self._session
        if session is None:
            return
        self._session = None
        public = session.public
        public.state, public.reason, public.expires_in_s = state, reason, 0.0
        public.expires_at_ns = _ns(min(session.expires_at, now))
        goal_state, goal_reason = {
            "closed": ("cancelled", "session_closed"),
            "expired": ("failed", "session_expired"),
        }.get(state, ("cancelled", reason))
        for record in self._goals.values():
            if record.goal.session_id == public.session_id and record.goal.state not in TERMINAL_GOAL_STATES:
                self._end_goal(record, _Ending(goal_state, goal_reason), now)
        for stream in self._streams.values():
            if stream.stream.session_id == public.session_id and stream.stream.state == "open":
                self._end_stream(stream, "interrupted", now, goal_reason)
        # Written after the stopping above is under way. The journal is the one record of a session that is
        # over: a repeated open or close reads it there, in this boot or a later one.
        self._journal.record(self.robot_id, public.session_id, public.model_dump(mode="json"), finished=True)
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
                contract = self._contract(request.goal)
                self._check_start(contract, self._clock.now())
                return Validation(resources=contract.resources, contract=contract)
            binding = find_binding(self._facts.bindings, request.stream.profile, request.stream.target, "stream")
            self._require_available(binding)
            return Validation(resources=binding.resources, limits=resolve_stream(request.stream, binding))

    def submit_goal(self, principal: Principal, request: GoalSubmit) -> Goal:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            body = request.model_dump(mode="json")
            prior = self._replay(principal, request.request_id, body, "goal")
            if prior is not None:
                # The same operation, never a second execution.
                return self._goal_view(prior.operation_id)
            self._authority(principal, request.session_id, request.context_id)
            contract = self._contract(request)
            self._check_start(contract, now)
            self._check_free(contract.resources)
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
            self._admit(principal, "goal", goal.goal_id, request_id=request.request_id, request=body, document=goal)
            self._goals[goal.goal_id] = _GoalRecord(
                goal,
                # Both deadlines start at admission; activation is recorded separately.
                wall_deadline=now + contract.max_wall_s,
                execution_deadline=self._feedback.execution_time_s + contract.max_execution_s,
            )
            for rid in contract.resources:
                self._resources[rid].status.activity = goal.goal_id
            self._history.emit("goal.accepted", now, goal_id=goal.goal_id, profile=contract.action.profile)
            return goal.model_copy(deep=True)

    def goal(self, goal_id: str) -> Goal:
        with self._lock:
            return self._goal_view(goal_id)

    def cancel_goal(self, principal: Principal, goal_id: str, authority: Authority) -> Goal:
        """Stop this goal only. Cancelling a finished goal changes nothing."""
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            record = self._goals.get(goal_id)
            if record is None or record.goal.state in TERMINAL_GOAL_STATES:
                return self._goal_view(goal_id)
            if not self._may_end(principal, authority) and record.goal.session_id != authority.session_id:
                raise RobotError("permission_denied", "this goal belongs to another session")
            self._end_goal(record, _Ending("cancelled", "cancelled"), now)
            return record.goal.model_copy(deep=True)

    def _goal_view(self, goal_id: str) -> Goal:
        record = self._goals.get(goal_id)
        if record is not None:
            return record.goal.model_copy(deep=True)
        return Goal.model_validate(self._stored(goal_id, "goal"))

    def _contract(self, goal: GoalSpec) -> Contract:
        """Resolve a proposed goal against its binding into the concrete command that would run."""
        binding = find_binding(self._facts.bindings, goal.action.profile, goal.action.target, "goal")
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
        action = resolve_action(goal.action, binding)
        # A goal that states how long its motion takes cannot succeed under a deadline that ends before the
        # motion and its settling do, so it is refused here instead of being moved and then failed. Execution
        # time never runs ahead of a realtime clock; the wall time of a stepped world says nothing about it.
        stated, settling = stated_duration_s(action), binding.limits.settle_time_s
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
        return Contract(
            action=action,
            resources=binding.resources,
            limits=binding.limits,
            stop_deadline_wall_s=binding.stop_deadline_wall_s,
            start_conditions=conditions,
            **deadlines,
        )

    def _check_start(self, contract: Contract, now: float) -> Action:
        """Run at admission and again at dispatch: nothing starts on unmeasured or changed state.

        Returns the action as it would start now; a trajectory is bound to the measured joint state.
        """
        self._require_measured(contract.resources)
        action = contract.action
        if isinstance(action, JointTrajectoryGoal):
            values = self._feedback.samples[contract.resources[0]].values
            action = activate_trajectory(action, contract.limits, values["positions"], values["velocities"])
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
                after = sample.values["positions"]
                moved = max((abs(after[name] - q) for name, q in before.values["positions"].items()), default=0.0)
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
        """Recheck start conditions, then dispatch. A rejection here causes no movement."""
        goal = record.goal
        contract = goal.contract
        try:
            if now >= record.wall_deadline:
                raise RobotError("start_condition_failed", "the wall deadline passed before dispatch")
            if self._feedback.execution_time_s >= record.execution_deadline:
                raise RobotError("start_condition_failed", "the execution deadline passed before dispatch")
            for rid in contract.resources:
                condition = self._resources[rid].status.condition
                if condition != "ready":
                    raise RobotError("start_condition_failed", f"resource {rid!r} became {condition}", resource=rid)
            contract.action = self._check_start(contract, now)
            goal.activated_time_ns = _ns(now)
            goal.state = "executing"
            # Durable before dispatch: after a crash this goal reads as possibly dispatched.
            self._journal.record(self.robot_id, goal.goal_id, goal.model_dump(mode="json"), finished=False)
            self._backend.apply(goal.goal_id, contract)
        except Exception as error:
            if not isinstance(error, RobotError):
                log.exception("backend rejected goal %s", goal.goal_id)
                error = RobotError("backend_error", str(error))
            goal.activated_time_ns = None
            for rid in contract.resources:
                # Release the reservation; what the resource retained before is untouched.
                self._resources[rid].status.activity = None
            self._finish_goal(
                record,
                "failed",
                "precondition_changed",
                now,
                detail=error.message,
                measurements={"code": error.code, **error.details},
            )
            return
        for rid in contract.resources:
            status = self._resources[rid].status
            status.condition, status.retained = "active", None
        self._history.emit("goal.executing", now, goal_id=goal.goal_id)

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
                elif status.phase == "done":
                    # Success comes from the controller's measured completion, not from the command being sent.
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
                        measurements=status.measurements,
                        observation_id=self._issue(now).observation_id,
                    )
                else:
                    goal.state = status.phase

    def _end_goal(self, record: _GoalRecord, ending: _Ending, now: float) -> None:
        goal = record.goal
        if goal.state == "accepted":
            # Never dispatched: nothing moved, so there is nothing to stop.
            for rid in goal.contract.resources:
                self._resources[rid].status.activity = None
            self._finish_goal(record, ending.state, ending.reason, now, detail=ending.detail)
        elif goal.state in ("executing", "finishing"):
            record.ending = ending
            goal.state = "stopping"
            self._begin_stop(goal.contract.resources, now)
            self._history.emit("goal.stopping", now, goal_id=goal.goal_id, reason=ending.reason)
        # Already stopping: the first cause stands.

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
        self._journal.record(self.robot_id, goal.goal_id, goal.model_dump(mode="json"), finished=True)
        self._history.emit(f"goal.{state}", now, goal_id=goal.goal_id, reason=reason)

    # ------------------------------------------------------------------
    # Target streams: open -> stopping -> closed. They never report success.
    # ------------------------------------------------------------------

    def open_stream(self, principal: Principal, request: StreamOpen) -> Stream:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            body = request.model_dump(mode="json")
            prior = self._replay(principal, request.request_id, body, "stream")
            if prior is not None:
                return self._stream_view(prior.operation_id, now)
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
            self._admit(principal, "stream", stream_id, request_id=request.request_id, request=body, document=stream)
            ack = TargetAck(
                sequence=0, basis_observation_id=request.basis_observation_id, expires_at_ns=_ns(expires_at)
            )
            self._streams[stream_id] = _StreamRecord(
                stream,
                expires_at,
                basis_sequence,
                mailbox=request.initial_target,
                acks=OrderedDict({0: (digest(request.initial_target.model_dump(mode="json")), ack)}),
            )
            for rid in binding.resources:
                self._resources[rid].status.activity = stream_id
            self._history.emit("stream.opened", now, stream_id=stream_id, profile=request.profile)
            return self._stream_view(stream_id, now)

    def update_stream(self, principal: Principal, stream_id: str, request: StreamUpdate) -> StreamUpdateResult:
        with self._lock:
            now = self._clock.now()
            # Expiry runs first, under the same lock: a late packet cannot revive an expired stream.
            self._expire(now)
            record = self._streams.get(stream_id)
            if record is None:
                stored = self._stream_view(stream_id, now)
                raise RobotError("stream_closed", "this stream is closed", state=stored.state, closure=stored.closure)
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
                return StreamUpdateResult(ack=known[1], duplicate=True, stream=self._stream_view(stream_id, now))
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
            return StreamUpdateResult(ack=ack, duplicate=False, stream=self._stream_view(stream_id, now))

    def stream(self, stream_id: str) -> Stream:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            return self._stream_view(stream_id, now)

    def close_stream(self, principal: Principal, stream_id: str, authority: Authority) -> Stream:
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            record = self._streams.get(stream_id)
            if record is None or record.stream.state != "open":
                return self._stream_view(stream_id, now)
            if not self._may_end(principal, authority) and record.stream.session_id != authority.session_id:
                raise RobotError("permission_denied", "this stream belongs to another session")
            self._end_stream(record, "client_closed", now)
            return self._stream_view(stream_id, now)

    def _stream_view(self, stream_id: str, now: float) -> Stream:
        record = self._streams.get(stream_id)
        if record is None:
            return Stream.model_validate(self._stored(stream_id, "stream"))
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
            try:
                if first:
                    # Durable before the first dispatch: after a crash this stream reads as one that may have
                    # moved its resources. Later targets are not journalled; that would put a disk write in
                    # every control tick.
                    dispatched = self._stream_view(stream.stream_id, now)
                    dispatched.applied_sequence = stream.accepted_sequence
                    self._journal.record(
                        self.robot_id, stream.stream_id, dispatched.model_dump(mode="json"), finished=False
                    )
                self._backend.track(stream.stream_id, stream.target, record.mailbox, stream.limits)
            except Exception as error:
                log.exception("a stream target could not be dispatched")
                self._end_stream(record, "failed", now, f"backend_error: {error}")
                continue
            stream.applied_sequence, record.mailbox = stream.accepted_sequence, None
            if first:
                for rid in stream.resources:
                    status = self._resources[rid].status
                    status.condition, status.retained = "active", None

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
            record.ending = closure
            stream.state = "stopping"
            self._begin_stop(stream.resources, now)
            self._history.emit("stream.stopping", now, stream_id=stream.stream_id, closure=closure)

    def _close_stream(self, record: _StreamRecord, closure: str, now: float, detail: str | None = None) -> None:
        stream = record.stream
        stream.state, stream.closure = "closed", closure
        stream.detail = detail or stream.detail
        record.mailbox = None
        view = self._stream_view(stream.stream_id, now)
        self._journal.record(self.robot_id, stream.stream_id, view.model_dump(mode="json"), finished=True)
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
            operation = self._new_operation("stop", principal, now)
            self._operations[operation.operation_id] = operation
            for record in self._goals.values():
                if record.goal.state not in TERMINAL_GOAL_STATES:
                    self._end_goal(record, _Ending("cancelled", "stopped"), now)
            for stream in self._streams.values():
                if stream.stream.state == "open":
                    self._end_stream(stream, "interrupted", now, "stopped")
            # Every resource is awaited: one that is already faulted has no measured standstill to report,
            # so the stop ends as stop_unconfirmed instead of succeeding around it.
            self._awaited[operation.operation_id] = list(self._resources)
            self._begin_stop(list(self._resources), now)
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

    def _check_free(self, resources: Sequence[str]) -> None:
        """One goal or stream per resource group, and no queue of future motion."""
        for rid in resources:
            status = self._resources[rid].status
            if status.condition == "stopping" or status.activity is not None:
                what = "stopping" if status.condition == "stopping" else f"in use by {status.activity}"
                raise RobotError("resource_busy", f"resource {rid!r} is {what}", resource=rid, activity=status.activity)
        for other in self._resources.values():
            busy = other.status
            if busy.id in resources or (busy.activity is None and busy.condition != "stopping"):
                continue
            for rid in resources:
                if not any(rid in group and busy.id in group for group in self._facts.concurrent):
                    raise RobotError(
                        "resource_busy",
                        f"{rid!r} cannot run while {busy.id!r} is active: that controller combination is not validated",
                        resource=rid,
                        active=busy.id,
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
        with self._lock:
            now = self._clock.now()
            self._expire(now)
            if not principal.operator:
                raise RobotError("permission_denied", "management operations need operator permission")
            body = request.model_dump(mode="json")
            prior = self._replay(principal, request.request_id, body, "management")
            if prior is not None:
                return self._operation_view(prior.operation_id)
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
            self._admit(
                principal,
                "management",
                operation.operation_id,
                request_id=request.request_id,
                request=body,
                document=operation,
            )
            self._operations[operation.operation_id] = operation
            self._management.append((operation, request))
            self._history.emit(
                "management.requested", now, operation_id=operation.operation_id, operation=request.operation
            )
            return operation.model_copy(deep=True)

    def operation(self, operation_id: str) -> Operation:
        with self._lock:
            return self._operation_view(operation_id)

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

    def _operation_view(self, operation_id: str) -> Operation:
        operation = self._operations.get(operation_id)
        if operation is not None:
            return operation.model_copy(deep=True)
        return Operation.model_validate(self._stored(operation_id, "management"))

    def _run_management(self, now: float) -> None:
        pending, self._management = self._management, []
        for operation, request in pending:
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

        self._facts = self._backend.describe()
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
        self._journal.record(self.robot_id, operation.operation_id, operation.model_dump(mode="json"), finished=True)
        self._history.emit(
            f"operation.{operation.state}", now, operation_id=operation.operation_id, operation=operation.kind
        )

    # ------------------------------------------------------------------
    # Journal: request identity and stored results
    # ------------------------------------------------------------------

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
            context_id=self.context_id,
            request=request,
            document=document.model_dump(mode="json"),
        )

    def _stored(self, operation_id: str, kind: str) -> dict[str, Any]:
        """The journal's copy of a goal, stream, or operation this incarnation no longer holds."""
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
                if document["applied_sequence"] is None:
                    detail = "runtime_restart: no target had been dispatched"
                else:
                    detail = "runtime_restart: targets were dispatched; later sequence numbers were not recorded"
                document = {**document, "state": "closed", "closure": "interrupted", "detail": detail}
            elif entry.kind == "session":
                document = {**document, "state": "revoked", "reason": "runtime_restart", "expires_in_s": 0.0}
            elif entry.kind == "management":
                document = {**document, "state": "failed", "reason": "runtime_restart"}
            self._journal.record(self.robot_id, entry.operation_id, document, finished=True)
