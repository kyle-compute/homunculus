"""Typed calls to an homunculus service and the bookkeeping a client needs.

A Target records where a robot is served and which session, context, goal, and
stream this client is working with, so callers do not pass IDs around. The
client never decides authority, expiry, or completion; the service does. It
does not renew a session or re-send a rejected command on its own.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .models import (
    ERROR_STATUS,
    TERMINAL_GOAL_STATES,
    Action,
    Authority,
    BaseVelocityTarget,
    Event,
    Goal,
    GoalSpec,
    GoalSubmit,
    ManagementRequest,
    Observation,
    Operation,
    RobotDescription,
    RobotError,
    Session,
    SessionOpen,
    SessionRenew,
    StartConditions,
    Stream,
    StreamOpen,
    StreamSpec,
    StreamUpdate,
    StreamUpdateResult,
    ValidateRequest,
    Validation,
    VideoLink,
)

M = TypeVar("M", bound=BaseModel)
T = TypeVar("T")

IMAGE_SUFFIX = {"image/png": ".png", "image/jpeg": ".jpg", "application/x-npy": ".npy"}
# The service answers a stop from workers it keeps for that, whatever else it is serving. A stop that is
# not answered this soon was sent to a service that is stuck, and the caller should learn that quickly.
STOP_ANSWER_S = 5.0
# Requests the service remembers by request ID: sent again with the same ID, they are not carried out twice.
REPEATABLE = (GoalSubmit, ManagementRequest, SessionOpen)
REPAIR = "repair it, or remove it and add the targets again with `homunculus connect`"
# For a stop that the recorded session cannot carry out: none is recorded, or it has ended.
STOP_WITHOUT_SESSION = "to stop the robot without a session, use `homunculus stop --operator`"


class ClientError(Exception):
    """The request could not be made or answered: no target, no recorded session, or no service."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint


class WaitTimeout(ClientError):
    """The operation is still running on the service; only the wait ended."""


class RequestInterrupted(KeyboardInterrupt):
    """Ctrl-C while a request was unanswered. The interrupt goes on as one; it also says what the request may do."""

    def __init__(self, message: str, hint: str):
        super().__init__(message)
        self.message = message
        self.hint = hint


class Target(BaseModel):
    """One robot on one service, plus what this client last did with it."""

    name: str
    url: str
    robot_id: str
    token: str
    operator_token: str | None = None
    # The local service home this target was read from, if any.
    home: str | None = None
    session_id: str | None = None
    context_id: str | None = None
    # The session was taken over with the operator token, so it belongs to the operator principal.
    session_operator: bool = False
    goal_id: str | None = None
    operation_id: str | None = None
    # The last management request and the context it was addressed to, so that a retry repeats it exactly.
    managed_request_id: str | None = None
    managed_context_id: str | None = None
    stream_id: str | None = None
    stream_sequence: int = 0


# Where a target is and how to reach it. Every other field is what the client remembers having done there.
ACCESS = {"name", "url", "robot_id", "token", "operator_token", "home"}


def default_home() -> Path:
    return Path(os.environ.get("HOMUNCULUS_HOME") or "~/.homunculus").expanduser()


@contextmanager
def _exclusive(directory: Path) -> Iterator[None]:
    """Hold a directory, made if need be, for one process at a time. The hold ends here or with the process."""
    while True:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except FileNotFoundError:
            continue  # removed again by the command that held it
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            # While this waited, the holder may have removed the directory and another command made it
            # again. What is held here is then not the directory that the others take.
            try:
                current = os.stat(directory).st_ino
            except FileNotFoundError:
                continue
            if current == os.fstat(descriptor).st_ino:
                yield
                return
        finally:
            os.close(descriptor)


class TargetStore:
    """Named targets in <home>/targets.json. The file holds access tokens, so it is owner-only.

    The file is shared by every command that runs at the same time, so save() writes
    only what this store changed since it was loaded or last saved.
    """

    def __init__(self, home: Path | None = None):
        self.home = home or default_home()
        self.path = self.home / "targets.json"
        stored = self._read()
        self.default: str | None = stored.get("default")
        self.targets: dict[str, Target] = {}
        for name, entry in stored.get("targets", {}).items():
            try:
                self.targets[name] = Target.model_validate(entry)
            except ValidationError as error:
                first = error.errors()[0]
                field = ".".join(str(part) for part in first["loc"]) or "entry"
                raise ClientError(
                    f"{self.path}: target {name!r} cannot be used ({field}: {first['msg']})", hint=REPAIR
                ) from None
        self._synced = self._view()

    def _read(self) -> dict[str, Any]:
        try:
            stored = json.loads(self.path.read_text()) if self.path.exists() else {}
        except ValueError as error:
            raise ClientError(f"{self.path} is not valid JSON ({error})", hint=REPAIR) from None
        if not isinstance(stored, dict) or not isinstance(stored.get("targets", {}), dict):
            raise ClientError(f"{self.path} does not hold saved targets", hint=REPAIR)
        return stored

    def _view(self) -> dict[str, Any]:
        return {"default": self.default, "targets": {name: t.model_dump() for name, t in self.targets.items()}}

    def save(self) -> None:
        """Apply this store's changes to the file as it is now.

        Commands overlap: a session can be opened while `homunculus events --follow` runs. Writing this
        store's whole view back would undo what the other command saved, and lose that session. Two
        commands can also finish in the same instant, so reading the file and replacing it is one step
        that commands take in turn.
        """
        before, mine = self._synced, self._view()
        self._synced = mine
        if not mine["targets"] and not self.home.exists():
            return  # nothing to keep, and no place where anything was kept
        with _exclusive(self.home):
            stored = self._read()
            targets = {name: dict(entry) for name, entry in stored.get("targets", {}).items()}
            for name in before["targets"].keys() - mine["targets"].keys():
                targets.pop(name, None)  # removed here
            for name, target in mine["targets"].items():
                base = before["targets"].get(name)
                if base is None:
                    targets[name] = target  # added here
                elif name in targets:  # one that another command removed meanwhile stays removed
                    targets[name].update({field: value for field, value in target.items() if value != base[field]})
            default = mine["default"] if mine["default"] != before["default"] else stored.get("default")
            if default not in targets:
                default = None  # its target is gone; no other robot is chosen in its place
            if not targets:
                # Nothing to remember: leave no file, and no directory if this store was all it held.
                self.path.unlink(missing_ok=True)
                try:
                    self.home.rmdir()  # removes only an empty directory
                except FileNotFoundError:
                    pass  # another command removed it first
                except OSError as error:
                    if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):  # it holds something else: it stays
                        raise
                return
            document = {"default": default, "targets": targets}
            if document == stored:
                return
            scratch = self.path.with_suffix(f".{os.getpid()}.tmp")  # per process: two savers never share one
            descriptor = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w") as file:
                json.dump(document, file, indent=2)
            scratch.replace(self.path)

    def get(self, name: str | None = None) -> Target:
        """The named target; with no name, $HOMUNCULUS_TARGET's or the default. An empty name is unknown, not the
        default."""
        if name is None:
            name = os.environ.get("HOMUNCULUS_TARGET") or self.default
        if name is None and self.targets:
            raise ClientError(
                f"no target is selected (saved targets: {', '.join(sorted(self.targets))})",
                hint="choose the default with `homunculus use NAME`, or name one with --target",
            )
        if name is None:
            raise ClientError(
                "no target is selected",
                hint="start the local demo with `homunculus demo start`, or add a service with `homunculus connect`",
            )
        if name not in self.targets:
            known = ", ".join(sorted(self.targets)) or "none"
            raise ClientError(
                f"unknown target {name!r} (known targets: {known})", hint="list them with `homunculus targets`"
            )
        return self.targets[name]

    def connect(
        self, url: str, token: str, operator_token: str | None = None, home: Path | None = None
    ) -> list[Target]:
        """Ask a service which robots it serves and record one target per robot."""
        try:
            response = httpx.get(
                f"{url}/v1", headers={"Authorization": f"Bearer {token}"}, timeout=5.0, trust_env=False
            )
        except httpx.InvalidURL as error:
            raise ClientError(f"{url!r} is not a URL this client can use: {error}") from None
        except httpx.TransportError as error:
            raise ClientError(f"cannot reach {url}: {error}") from None
        if not response.is_success:
            raise _refusal(response)
        try:
            robot_ids = [str(robot["robot_id"]) for robot in response.json()["robots"]]
        except (ValueError, KeyError, TypeError):
            raise _another_program(response) from None
        added = []
        for robot_id in robot_ids:
            # "fake-arm" is addressed as "fake:arm": the prefix says what kind of robot it is.
            name = robot_id.replace("-", ":", 1)
            target = Target(
                name=name,
                url=url,
                robot_id=robot_id,
                token=token,
                operator_token=operator_token,
                home=str(home) if home else None,
            )
            known = self.targets.get(name)
            if known is not None and (known.url, known.robot_id) == (url, robot_id):
                # The same robot on the same service: its access is replaced, and what was done there
                # (the session above all) stays known.
                target = target.model_copy(update=known.model_dump(exclude=ACCESS))
            self.targets[name] = target
            added.append(target)
        if added and self.default not in self.targets:
            self.default = added[0].name
        return added

    def connect_home(self, home: Path) -> list[Target]:
        """Record the robots of a service running on this machine, from its home directory."""
        record, tokens = home / "service.json", home / "tokens.json"
        if not record.exists() or not tokens.exists():
            raise ClientError(
                f"no running service found in {home}", hint="start one with `homunculus serve --home ...`"
            )
        try:
            url = json.loads(record.read_text())["url"]
            access = json.loads(tokens.read_text())
            client, operator = access["client"], access["operator"]
        except (ValueError, KeyError, TypeError):
            raise ClientError(
                f"the service record or the tokens in {home} cannot be read",
                hint="restart the service that runs there (`homunculus serve --home ...`); it says what is wrong",
            ) from None
        return self.connect(url, client, operator, home)

    def forget(self, names: Sequence[str]) -> list[str]:
        """Drop saved targets. Nothing is sent to their services."""
        removed = [name for name in names if self.targets.pop(name, None) is not None]
        if self.default in removed:
            self.default = None  # no other robot is chosen in its place: the next command says which it means
        return removed

    def forget_home(self, home: Path) -> list[str]:
        return self.forget([name for name, target in self.targets.items() if target.home == str(home)])


class RobotClient:
    def __init__(
        self,
        target: Target,
        *,
        timeout_s: float = 15.0,
        retries: int = 2,
        on_change: Callable[[], None] | None = None,
    ):
        """on_change is called each time the client updates what the target remembers (session, last goal,
        stream, operation). The CLI saves the targets there, so that a command which is still waiting for
        its goal has already recorded it for the next command."""
        self.target = target
        self._on_change = on_change
        self._timeout_s = timeout_s
        self._retries = retries
        self._base = f"{target.url}/v1/robots/{target.robot_id}"
        # Straight to the address the target names. A proxy that the environment names for other programs
        # (HTTP_PROXY, the system's settings) would be sent the robot's commands and tokens, and a stop
        # would depend on it.
        self._http = httpx.Client(timeout=timeout_s, trust_env=False)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> RobotClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reading ----------------------------------------------------------

    def describe(self) -> RobotDescription:
        return self._call("GET", "", RobotDescription)

    def observe(
        self, cameras: Sequence[str] = (), after_sequence: int | None = None, wait_s: float = 0.0
    ) -> Observation:
        params: dict[str, Any] = {"camera": list(cameras)}
        if after_sequence is not None:
            params.update(after_sequence=after_sequence, wait_s=wait_s)
        return self._call("GET", "/observations", Observation, params=params)

    def observation(self, observation_id: str) -> Observation:
        return self._call("GET", f"/observations/{observation_id}", Observation)

    def image(self, handle: str) -> bytes:
        return self._send("GET", f"/images/{handle}").content

    def save_images(self, observation: Observation, directory: Path) -> list[Path]:
        """Download an observation's images, named by sensor, kind, and handle."""
        directory.mkdir(parents=True, exist_ok=True)
        saved = []
        for image in observation.images:
            suffix = IMAGE_SUFFIX.get(image.media_type, ".bin")
            path = directory / f"{image.sensor}-{image.kind}-{image.handle}{suffix}"
            path.write_bytes(self.image(image.handle))
            saved.append(path)
        return saved

    def video_link(self, camera: str) -> VideoLink:
        """A link that plays the camera live in a browser, which opens it without this client's token."""
        return self._call("GET", "/video-link", VideoLink, params={"camera": camera})

    def events(self, after: int = 0, follow: bool = False) -> Iterator[Event]:
        """Numbered events. With follow, reconnect after a dropped connection from the last event seen.

        A gap is reported as an `events.dropped` event, never skipped silently. Numbering starts over when the
        service restarts, so a reconnection that finds a new boot reports a gap and reads that boot from its
        first event.
        """
        last, boot_id = after, None
        connected = False
        while True:
            try:
                with self._http.stream(
                    "GET",
                    f"{self._base}/events",
                    params={"after": last, "follow": follow},
                    headers=self._headers(),
                    # Only a reader that follows waits without limit: its stream is silent until an event.
                    timeout=httpx.Timeout(self._timeout_s, read=None if follow else self._timeout_s),
                ) as response:
                    if not response.is_success:
                        response.read()
                        raise _refusal(response)
                    connected = True
                    serving = response.headers.get("robot-boot-id")
                    if boot_id not in (None, serving):
                        gap = {"after": last, "reason": "runtime_restart"}
                        yield Event(sequence=last, time_ns="0", type="events.dropped", data=gap)
                        last, boot_id = 0, serving
                        continue  # ask again, from the new boot's first event
                    boot_id = serving
                    kind = ""
                    for line in response.iter_lines():
                        if line.startswith("event:"):
                            kind = line[6:].strip()
                        elif line.startswith("data:") and kind == "dropped":
                            yield Event(sequence=last, time_ns="0", type="events.dropped", data=json.loads(line[5:]))
                            last = 0  # as the service does: the events that follow set the cursor again
                        elif line.startswith("data:"):
                            event = Event.model_validate_json(line[5:])
                            last = event.sequence
                            yield event
            except httpx.TransportError as error:
                if not (follow and connected):  # only a reader that was following reconnects
                    raise self._unreachable(error, self._timeout_s) from None
                time.sleep(0.5)
                continue
            if not follow:
                return

    # -- authority --------------------------------------------------------

    def open_session(
        self, lifetime_wall_s: float | None = None, takeover: bool = False, request_id: str | None = None
    ) -> Session:
        """Read the robot's current context and acquire the control session in it.

        Pass the request_id of an attempt that got no answer to learn its session instead of being
        refused by it.
        """
        request = SessionOpen(
            request_id=_stated(request_id, _request_id),
            context_id=self.describe().context_id,
            lifetime_wall_s=lifetime_wall_s,
            takeover=takeover,
        )
        session = self._call("POST", "/sessions", Session, body=request, operator=takeover)
        self._record(
            session_id=session.session_id,
            context_id=session.context_id,
            session_operator=takeover,
            stream_id=None,
            stream_sequence=0,
        )
        return session

    def renew_session(self, lifetime_wall_s: float | None = None) -> Session:
        request = SessionRenew(lifetime_wall_s=lifetime_wall_s)
        return self._call("POST", f"/sessions/{self._session()}/renew", Session, body=request)

    def close_session(self) -> Session:
        session = self._call("POST", f"/sessions/{self._session()}/close", Session)
        self._record(session_id=None, stream_id=None, session_operator=False)
        return session

    # -- finite goals -----------------------------------------------------

    def validate(self, goal: GoalSpec | None = None, stream: StreamSpec | None = None) -> Validation:
        """Check a goal or a stream as it would be submitted. Nothing is reserved or sent to the robot."""
        return self._call("POST", "/validate", Validation, body=ValidateRequest(goal=goal, stream=stream))

    def submit_goal(
        self,
        action: Action,
        *,
        request_id: str | None = None,
        max_execution_s: float | None = None,
        max_wall_s: float | None = None,
        start_conditions: StartConditions | None = None,
        session_id: str | None = None,
        context_id: str | None = None,
    ) -> Goal:
        """Submit one finite goal. Pass the same request_id to retry it without a second execution.

        The goal is addressed to the recorded session and context. A request written for another session
        or context is sent as written, and the service judges whether it still applies.
        """
        request = GoalSubmit(
            request_id=_stated(request_id, _request_id),
            session_id=_stated(session_id, self._session),
            context_id=_stated(context_id, lambda: self.target.context_id),
            max_execution_s=max_execution_s,
            max_wall_s=max_wall_s,
            action=action,
            start_conditions=start_conditions,
        )
        goal = self._call("POST", "/goals", Goal, body=request)
        self._record(goal_id=goal.goal_id)
        return goal

    def goal(self, goal_id: str | None = None) -> Goal:
        return self._call("GET", f"/goals/{self._known(goal_id, self.target.goal_id, 'goal')}", Goal)

    def cancel_goal(self, goal_id: str | None = None) -> Goal:
        path = f"/goals/{self._known(goal_id, self.target.goal_id, 'goal')}/cancel"
        return self._call("POST", path, Goal, body=self._authority())

    def wait_goal(self, goal_id: str | None = None, timeout_s: float = 60.0) -> Goal:
        """Poll until the service reports the goal finished. The outcome is the service's, not a guess."""
        deadline = time.monotonic() + timeout_s
        while True:
            goal = self.goal(goal_id)
            if goal.state in TERMINAL_GOAL_STATES:
                return goal
            if time.monotonic() >= deadline:
                raise WaitTimeout(
                    f"goal {goal.goal_id} is still {goal.state} after {timeout_s:g} s",
                    hint=(
                        "only the wait ended; see `homunculus goal inspect`, "
                        "or end the goal with `homunculus goal cancel`"
                    ),
                )
            time.sleep(0.05)

    # -- target streams ---------------------------------------------------

    def open_stream(
        self,
        profile: str,
        target: str,
        initial_target: BaseVelocityTarget,
        update_validity_wall_s: float,
        *,
        basis_observation_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        context_id: str | None = None,
    ) -> Stream:
        """Open a stream with its first target.

        The target is valid for the stream's window from the issue time of the observation it is based on.
        Name the observation the target was decided on; without one, an observation is read now.
        """
        request = StreamOpen(
            request_id=_stated(request_id, _request_id),
            session_id=_stated(session_id, self._session),
            context_id=_stated(context_id, lambda: self.target.context_id),
            profile=profile,
            target=target,
            update_validity_wall_s=update_validity_wall_s,
            initial_target=initial_target,
            basis_observation_id=_stated(basis_observation_id, lambda: self.observe().observation_id),
        )
        stream = self._call("POST", "/streams", Stream, body=request)
        self._record(stream_id=stream.stream_id, stream_sequence=stream.accepted_sequence)
        return stream

    def update_stream(
        self,
        setpoint: BaseVelocityTarget,
        *,
        basis_observation_id: str | None = None,
        stream_id: str | None = None,
        sequence: int | None = None,
        session_id: str | None = None,
        context_id: str | None = None,
    ) -> StreamUpdateResult:
        """Replace the stream's target, as the next sequence number unless one is given.

        Name the observation the target was decided on: the target is valid for the stream's window from
        that observation's issue time, however long deciding took. Without one, an observation is read now.
        """
        stream_id = self._known(stream_id, self.target.stream_id, "stream")
        request = StreamUpdate(
            session_id=_stated(session_id, self._session),
            context_id=_stated(context_id, lambda: self.target.context_id),
            sequence=_stated(sequence, lambda: self.target.stream_sequence + 1),
            basis_observation_id=_stated(basis_observation_id, lambda: self.observe().observation_id),
            target=setpoint,
        )
        result = self._call("PUT", f"/streams/{stream_id}/target", StreamUpdateResult, body=request)
        self._record(stream_sequence=result.stream.accepted_sequence)
        return result

    def stream(self, stream_id: str | None = None) -> Stream:
        return self._call("GET", f"/streams/{self._known(stream_id, self.target.stream_id, 'stream')}", Stream)

    def close_stream(self, stream_id: str | None = None) -> Stream:
        path = f"/streams/{self._known(stream_id, self.target.stream_id, 'stream')}/close"
        return self._call("POST", path, Stream, body=self._authority())

    # -- stop and management ----------------------------------------------

    def stop(self, as_operator: bool = False) -> Operation:
        """Stop everything. An operator stop needs no session and works across context changes."""
        if not as_operator and self.target.session_id is None:
            raise ClientError(f"no control session is recorded for {self.target.name}", hint=STOP_WITHOUT_SESSION)
        authority = Authority() if as_operator else self._authority()
        operator = True if as_operator else None
        patience = min(self._timeout_s, STOP_ANSWER_S)
        operation = self._call("POST", "/stop", Operation, body=authority, operator=operator, timeout_s=patience)
        self._record(operation_id=operation.operation_id)
        return operation

    def manage(
        self, operation: str, parameters: dict[str, Any] | None = None, request_id: str | None = None
    ) -> Operation:
        request_id = _stated(request_id, _request_id)
        if request_id != self.target.managed_request_id:
            # A new request is addressed to the context as it is now. A retry must repeat the first attempt
            # exactly, and an operation that succeeded has usually changed the context since.
            self._record(managed_request_id=request_id, managed_context_id=self.describe().context_id)
        request = ManagementRequest(
            request_id=request_id,
            context_id=self.target.managed_context_id,
            operation=operation,
            parameters=parameters or {},
        )
        started = self._call("POST", "/management", Operation, body=request, operator=True)
        self._record(operation_id=started.operation_id)
        return started

    def operation(self, operation_id: str | None = None) -> Operation:
        operation_id = self._known(operation_id, self.target.operation_id, "operation")
        return self._call("GET", f"/operations/{operation_id}", Operation)

    def wait_operation(self, operation_id: str | None = None, timeout_s: float = 60.0) -> Operation:
        deadline = time.monotonic() + timeout_s
        while True:
            operation = self.operation(operation_id)
            if operation.state != "running":
                return operation
            if time.monotonic() >= deadline:
                raise WaitTimeout(
                    f"operation {operation.operation_id} is still running after {timeout_s:g} s",
                    hint="only the wait ended; see `homunculus operation inspect`",
                )
            time.sleep(0.05)

    # -- transport --------------------------------------------------------

    def _record(self, **fields: Any) -> None:
        """Update what the target remembers, and tell whoever saves it."""
        for field, value in fields.items():
            setattr(self.target, field, value)
        if self._on_change is not None:
            self._on_change()

    def _session(self) -> str:
        if self.target.session_id is None:
            raise ClientError(
                f"no control session is recorded for {self.target.name}", hint="open one with `homunculus session open`"
            )
        return self.target.session_id

    def _authority(self) -> Authority:
        return Authority(session_id=self._session(), context_id=self.target.context_id)

    def _known(self, given: str | None, recorded: str | None, what: str) -> str:
        """The ID the caller gave, as given (an empty one too: the service refuses it); the one this target
        recorded only when the caller gave none."""
        value = recorded if given is None else given
        if value is None:
            raise ClientError(f"no {what} is recorded for {self.target.name}; pass its id")
        return value

    def _headers(self, operator: bool | None = None) -> dict[str, str]:
        """Who is asking. Unless the call says, it is whoever owns the recorded session: the service binds
        a session to the principal that opened it, and a takeover is opened as operator."""
        if operator is None:
            operator = self.target.session_operator
        token = self.target.operator_token if operator else self.target.token
        if token is None:
            raise ClientError(
                f"target {self.target.name} has no operator token, which this operation needs",
                hint="reconnect with `homunculus connect URL --token ... --operator-token ...`",
            )
        return {"Authorization": f"Bearer {token}"}

    def _call(
        self,
        method: str,
        path: str,
        result: type[M],
        *,
        body: BaseModel | None = None,
        params: dict[str, Any] | None = None,
        operator: bool | None = None,
        timeout_s: float | None = None,
    ) -> M:
        response = self._send(method, path, body=body, params=params, operator=operator, timeout_s=timeout_s)
        try:
            return result.model_validate_json(response.content)
        except ValidationError as error:
            first = error.errors()[0]
            field = ".".join(str(part) for part in first["loc"]) or "document"
            raise ClientError(
                f"{self.target.url} answered {method} {path or '/'} in a form this client does not read "
                f"({field}: {first['msg']})",
                hint="the service may be another version; it did act on the request, so look before repeating it",
            ) from None

    def _send(
        self,
        method: str,
        path: str,
        *,
        body: BaseModel | None = None,
        params: dict[str, Any] | None = None,
        operator: bool | None = None,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        """Send one request, repeating it unchanged if the connection fails.

        Repeating is safe: finite mutations carry a request_id, stream updates a
        sequence number, and the remaining calls do not start motion. A request
        that is not answered in time is not repeated: a service that is busy or
        stuck gains nothing from a second copy, and the caller would wait longer
        to hear of it.
        """
        content = body.model_dump_json(exclude_none=True) if body is not None else None
        headers = self._headers(operator)
        if content is not None:
            headers["Content-Type"] = "application/json"
        patience = self._timeout_s if timeout_s is None else timeout_s
        attempt = 0
        while True:
            try:
                response = self._http.request(
                    method, self._base + path, content=content, params=params, headers=headers, timeout=patience
                )
            except KeyboardInterrupt:
                # Giving up on the answer does not take the request back: the service may act on it later.
                if method == "GET":
                    hint = "the unanswered request was a read; it commands nothing"
                else:
                    hint = "the request may still have been received; check the service" + self._to_repeat(body)
                raise RequestInterrupted(f"interrupted before {self.target.url} answered", hint) from None
            except httpx.TransportError as error:
                if attempt == self._retries or isinstance(error, httpx.TimeoutException):
                    raise self._unreachable(error, patience, body) from None
                attempt += 1
                time.sleep(0.2 * attempt)
                continue
            if response.is_success:
                return response
            raise _refusal(response)

    def _unreachable(self, error: httpx.TransportError, patience: float, body: BaseModel | None = None) -> ClientError:
        if isinstance(error, httpx.TimeoutException):
            message = f"{self.target.url} did not answer within {patience:g} s"
            hint = "the request may still have been received; check the service"
        else:
            message = f"cannot reach {self.target.url}: {error}"
            hint = "check that the service is running"
        return ClientError(message, hint=hint + self._to_repeat(body))

    @staticmethod
    def _to_repeat(body: BaseModel | None) -> str:
        """The rest of a hint about a request that got no answer: where to look, and how to send it again."""
        rest = " (for the local demo: `homunculus demo status`)"
        if isinstance(body, REPEATABLE):
            # Without this ID a second attempt would be a second request, and could move the robot twice.
            rest += f"; to send this request again and not run it twice, repeat it with --request-id {body.request_id}"
        return rest


def _request_id() -> str:
    return f"req-{uuid.uuid4().hex}"


def _stated(given: T | None, otherwise: Callable[[], T]) -> T:
    """What the caller stated, as stated; what this client knows only when the caller left it out.

    An empty request ID is not left out. Used in place of a new one, it would let a repeated command
    run a second time; sent as written, it is refused.
    """
    return otherwise() if given is None else given


def _another_program(response: httpx.Response) -> ClientError:
    origin = f"{response.url.scheme}://{response.url.netloc.decode('ascii')}"
    return ClientError(
        f"{origin} answered (HTTP {response.status_code}), but not as an homunculus service",
        hint="check the address and the port; another program may be listening there",
    )


def _refusal(response: httpx.Response) -> RobotError | ClientError:
    """The service's structured error, as the same exception type the runtime raises."""
    try:
        info = response.json()["error"]
        code, message, details = info["code"], info["message"], info.get("details", {})
    except (ValueError, KeyError, TypeError):
        return _another_program(response)  # the service gives every refusal that shape, whatever the path
    if code not in ERROR_STATUS:
        code, message = "internal_error", f"{code}: {message}"
    error = RobotError(code, message)
    error.details = details
    return error
