"""HTTP and event transport for the runtime: access checks, error mapping, and startup.

Each handler turns a request into one runtime call and turns its result, or its
RobotError, into a response. Nothing here decides whether a robot may move.
serve() wires the concrete backend to a runtime and owns the control threads.
"""

import asyncio
import contextlib
import hmac
import importlib
import inspect
import json
import logging
import math
import os
import secrets
import signal
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Self

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .journal import Journal, JournalInUse
from .models import (
    ERROR_STATUS,
    Authority,
    Goal,
    GoalSubmit,
    ManagementRequest,
    Observation,
    Operation,
    RobotDescription,
    RobotError,
    RobotSummary,
    ServiceInfo,
    ServiceLimits,
    Session,
    SessionOpen,
    SessionRenew,
    Stream,
    StreamOpen,
    StreamUpdate,
    StreamUpdateResult,
    ValidateRequest,
    Validation,
    VideoLink,
)
from .rigs import NEEDS_MUJOCO
from .runtime import Backend, Frame, Principal, Runtime, StartupError

log = logging.getLogger(__name__)

# `--backend NAME` for the backends that ship with homunculus. Any other backend is named by its import path.
BUILT_IN = {
    "fake": "homunculus.backends.fake:robots",
    "mujoco": "homunculus.backends.mujoco:robots",
    "task": "homunculus.backends.task:robots",
}


class Service:
    """The runtimes one process serves, their control and stopping threads, and who may call them.

    The control threads run for the length of a `with` block. Leaving it stops them and then the robots.
    """

    def __init__(self, runtimes: dict[str, Runtime], tokens: dict[str, Principal], boot_id: str, limits: ServiceLimits):
        self.runtimes = runtimes
        self.tokens = tokens
        self.boot_id = boot_id
        self.limits = limits
        self.closing = threading.Event()
        self.subscribers = 0
        self.viewers = 0
        # Signs video links, which a browser opens without the token. A key of this boot's own ends every link
        # of the boot before.
        self._video_key = secrets.token_bytes(32)
        self._threads = [
            threading.Thread(target=self._control, args=(runtime,), name=f"control:{rid}", daemon=True)
            for rid, runtime in runtimes.items()
        ]
        # A request that ends motion runs on workers of its robot's own. It then waits neither for one of
        # the workers the other requests share nor behind what is being stopped on another robot.
        self.stop_workers = {
            rid: ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"stop:{rid}") for rid in runtimes
        }

    def __enter__(self) -> Self:
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.closing.set()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        for workers in self.stop_workers.values():
            workers.shutdown(wait=False, cancel_futures=True)
        for runtime in self.runtimes.values():
            runtime.close()

    def _control(self, runtime: Runtime) -> None:
        """One thread per robot calls tick() at the backend's period. Nothing else commands the backend."""
        due = time.monotonic()
        while not self.closing.is_set():
            try:
                runtime.tick()
            except Exception:
                log.exception("control tick failed for %s", runtime.robot_id)
            due += runtime.period_s
            wait = due - time.monotonic()
            if wait > 0:
                self.closing.wait(wait)
            else:
                due = time.monotonic()  # overran: resume from now instead of ticking in a burst

    def principal(self, authorization: str | None) -> Principal:
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() == "bearer":
            for token, principal in self.tokens.items():
                if hmac.compare_digest(token.encode(), presented.strip().encode()):
                    return principal
        raise RobotError("unauthenticated", "send a valid token as 'Authorization: Bearer <token>'")

    def runtime(self, robot_id: str) -> Runtime:
        if robot_id not in self.runtimes:
            raise RobotError("unknown_robot", f"no robot named {robot_id!r} here", robots=sorted(self.runtimes))
        return self.runtimes[robot_id]

    def summaries(self) -> list[RobotSummary]:
        return [
            RobotSummary(robot_id=rid, backend=runtime.describe().backend) for rid, runtime in self.runtimes.items()
        ]

    def video_signature(self, robot_id: str, camera: str, until: int) -> str:
        """What makes a video link valid for one camera until a time: only this boot of the service computes it."""
        return hmac.new(self._video_key, json.dumps([robot_id, camera, until]).encode(), "sha256").hexdigest()


def _refusal(error: RobotError) -> JSONResponse:
    return JSONResponse({"error": error.info().model_dump(mode="json")}, status_code=ERROR_STATUS[error.code])


def _bound_requests(app: ASGIApp, maximum: int) -> ASGIApp:
    """Refuse an oversized request before any of its body is read.

    This sits in front of the framework, which buffers a whole body before a handler or dependency sees the
    request. A declared length is compared with the limit. A body sent without one (chunked) could only be
    measured by reading it, so it is refused whatever its size.
    """

    async def bounded(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            length = headers.get(b"content-length", b"0")
            error = None
            if b"transfer-encoding" in headers or not length.isdigit():
                error = RobotError(
                    "invalid_request",
                    f"send the request body with a Content-Length; bodies are limited to {maximum} bytes and "
                    "one without a declared length is not read",
                    maximum=maximum,
                )
            elif int(length) > maximum:
                error = RobotError(
                    "request_too_large", f"request bodies are limited to {maximum} bytes", maximum=maximum
                )
            if error is not None:
                await _refusal(error)(scope, receive, send)
                return
        await app(scope, receive, send)

    return bounded


def create_app(service: Service) -> FastAPI:
    # Two paths need no token: /healthz and the generated description, /openapi.json. The pages that
    # browse that description are not served: they would load scripts from a CDN into this origin. Live
    # video takes a signed link in place of the token.
    app = FastAPI(title="homunculus", version=__version__, docs_url=None, redoc_url=None)
    app.add_middleware(_bound_requests, maximum=service.limits.max_request_bytes)

    # Handlers that are plain functions run on a pool of workers that all requests share. The access
    # checks and the error responses are coroutines instead: they run on the event loop, so a request
    # that ends motion needs no shared worker at any point, whether it is accepted or refused.
    async def caller(authorization: Annotated[str | None, Header()] = None) -> Principal:
        return service.principal(authorization)

    async def addressed(robot_id: str) -> Runtime:
        return service.runtime(robot_id)

    Caller = Annotated[Principal, Depends(caller)]
    Robot = Annotated[Runtime, Depends(addressed)]

    async def ending(runtime: Runtime, call: Callable[..., Any], *arguments: Any) -> Any:
        """Run a runtime call that ends motion on the robot's own stopping workers.

        An observation keeps a shared worker for as long as its cameras take, and enough of them at once
        occupy every one. A stop, a cancellation, or a closure must not wait for a worker to come free.
        """
        workers = service.stop_workers[runtime.robot_id]
        return await asyncio.get_running_loop().run_in_executor(workers, call, *arguments)

    @app.exception_handler(RobotError)
    async def refused(request: Request, error: RobotError) -> JSONResponse:
        return _refusal(error)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        return _refusal(_invalid_request(error.errors()))

    @app.exception_handler(HTTPException)
    async def no_route(request: Request, error: HTTPException) -> JSONResponse:
        code = "not_found" if error.status_code == 404 else "invalid_request"
        message = f"{request.method} {request.url.path}: {error.detail}"
        return JSONResponse({"error": {"code": code, "message": message, "details": {}}}, error.status_code)

    @app.exception_handler(Exception)
    async def crashed(request: Request, error: Exception) -> JSONResponse:
        log.exception("unhandled error serving %s %s", request.method, request.url.path)
        return _refusal(RobotError("internal_error", "the service failed to handle this request"))

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "boot_id": service.boot_id}

    @app.get("/v1")
    def info(principal: Caller) -> ServiceInfo:
        return ServiceInfo(version=__version__, boot_id=service.boot_id, robots=service.summaries())

    @app.get("/v1/robots/{robot_id}/video", name="video")
    async def video(robot_id: str, camera: str, until: int, signature: str) -> StreamingResponse:
        """A camera's live video: one response whose parts are images, which a browser shows in turn.

        A browser opens it without the token, so it is not on the router below, which asks for one. The
        signature from /video-link lets it in instead, for the robot, camera, and time it was made for.
        """
        if not hmac.compare_digest(signature.encode(), service.video_signature(robot_id, camera, until).encode()):
            raise RobotError(
                "unauthenticated",
                "this video link was not made by this service, or the service restarted since; "
                "get a new one with `homunculus video CAMERA`",
            )
        if time.time() > until:
            raise RobotError(
                "unauthenticated", "this video link has expired; get a new one with `homunculus video CAMERA`"
            )
        runtime = service.runtime(robot_id)
        if service.viewers >= service.limits.max_video_viewers:
            raise RobotError(
                "too_many_subscribers",
                "the live video viewer limit is reached; close another video or try again later",
                maximum=service.limits.max_video_viewers,
            )
        # Counted here, on the event loop, where no other request runs between the check and the count. The
        # place is given back when the video ends, or at once if it does not begin.
        service.viewers += 1
        try:
            # Off the event loop, as every runtime call here: they wait for the runtime's lock. The first frame
            # is captured before the response begins, so a camera that cannot be shown is refused with the
            # reason instead of playing as a video with no frames.
            period = 1 / await asyncio.to_thread(runtime.video_fps, camera)
            started = time.monotonic()
            first = await asyncio.to_thread(runtime.video_frame, camera)
        except BaseException:
            service.viewers -= 1
            raise

        async def frames(frame: Frame, started: float) -> AsyncIterator[bytes]:
            try:
                while True:
                    head = (
                        f"--frame\r\nContent-Type: {frame.image.media_type}\r\n"
                        f"Content-Length: {len(frame.data)}\r\n\r\n"
                    )
                    yield head.encode() + frame.data + b"\r\n"
                    await asyncio.sleep(started + period - time.monotonic())
                    started = time.monotonic()
                    try:
                        frame = await asyncio.to_thread(runtime.video_frame, camera)
                    except RobotError as error:
                        # The response has begun, so its status can no longer say why: the video ends here.
                        log.warning("live video of %s on %s ended: %s", camera, robot_id, error.message)
                        return
            finally:
                service.viewers -= 1

        # A viewer that leaves ends the response: the framework then cancels frames() wherever it waits.
        media_type = "multipart/x-mixed-replace; boundary=frame"
        return StreamingResponse(frames(first, started), media_type=media_type, headers={"Cache-Control": "no-store"})

    robot = APIRouter(prefix="/v1/robots/{robot_id}", dependencies=[Depends(caller)])

    @robot.get("")
    def describe(runtime: Robot) -> RobotDescription:
        return runtime.describe()

    @robot.get("/observations")
    def observe(
        runtime: Robot,
        camera: Annotated[list[str] | None, Query()] = None,
        after_sequence: int | None = None,
        wait_s: Annotated[float, Query(ge=0)] = 0.0,
    ) -> Observation:
        # A bounded wait for a reading newer than the caller's last observation. The bound is the service's:
        # a longer wait, or one without limit, is cut to it. A wait that is not a number of seconds is refused.
        deadline = time.monotonic() + min(wait_s, service.limits.observation_wait_max_s)
        while True:
            try:
                return runtime.observe(camera or (), after_sequence)
            except RobotError as error:
                if error.code != "no_new_observation" or time.monotonic() >= deadline:
                    raise
            time.sleep(min(runtime.period_s, 0.05))

    @robot.get("/observations/{observation_id}")
    def observation(runtime: Robot, observation_id: str) -> Observation:
        return runtime.observation(observation_id)

    @robot.get("/images/{handle}")
    def image(runtime: Robot, handle: str) -> Response:
        media_type, data = runtime.image(handle)
        return Response(content=data, media_type=media_type)

    @robot.get("/video-link")
    def video_link(runtime: Robot, request: Request, camera: str) -> VideoLink:
        fps = runtime.video_fps(camera)
        lifetime = service.limits.video_link_lifetime_s
        until = math.ceil(time.time() + lifetime)  # rounded up: the link lasts at least as long as it says
        signature = service.video_signature(runtime.robot_id, camera, until)
        # On the address this request came to: the one the caller reaches the service at.
        url = request.url_for("video", robot_id=runtime.robot_id)
        url = url.include_query_params(camera=camera, until=until, signature=signature)
        return VideoLink(camera=camera, fps=fps, url=str(url), expires_in_s=lifetime)

    @robot.post("/sessions", status_code=201)
    def open_session(runtime: Robot, principal: Caller, request: SessionOpen) -> Session:
        return runtime.open_session(principal, request)

    @robot.post("/sessions/{session_id}/renew")
    def renew_session(runtime: Robot, principal: Caller, session_id: str, request: SessionRenew) -> Session:
        return runtime.renew_session(principal, session_id, request)

    @robot.post("/sessions/{session_id}/close")
    async def close_session(runtime: Robot, principal: Caller, session_id: str) -> Session:
        return await ending(runtime, runtime.close_session, principal, session_id)

    @robot.post("/validate")
    def validate(runtime: Robot, request: ValidateRequest) -> Validation:
        return runtime.validate(request)

    @robot.post("/goals", status_code=202)
    def submit_goal(runtime: Robot, principal: Caller, request: GoalSubmit) -> Goal:
        return runtime.submit_goal(principal, request)

    @robot.get("/goals/{goal_id}")
    def goal(runtime: Robot, goal_id: str) -> Goal:
        return runtime.goal(goal_id)

    @robot.post("/goals/{goal_id}/cancel", status_code=202)
    async def cancel_goal(runtime: Robot, principal: Caller, goal_id: str, authority: Authority | None = None) -> Goal:
        return await ending(runtime, runtime.cancel_goal, principal, goal_id, authority or Authority())

    @robot.post("/streams", status_code=201)
    def open_stream(runtime: Robot, principal: Caller, request: StreamOpen) -> Stream:
        return runtime.open_stream(principal, request)

    @robot.put("/streams/{stream_id}/target")
    def update_stream(runtime: Robot, principal: Caller, stream_id: str, request: StreamUpdate) -> StreamUpdateResult:
        return runtime.update_stream(principal, stream_id, request)

    @robot.get("/streams/{stream_id}")
    def stream(runtime: Robot, stream_id: str) -> Stream:
        return runtime.stream(stream_id)

    @robot.post("/streams/{stream_id}/close", status_code=202)
    async def close_stream(
        runtime: Robot, principal: Caller, stream_id: str, authority: Authority | None = None
    ) -> Stream:
        return await ending(runtime, runtime.close_stream, principal, stream_id, authority or Authority())

    @robot.post("/stop", status_code=202)
    async def stop(runtime: Robot, principal: Caller, authority: Authority | None = None) -> Operation:
        # An operator who stops outside a session has no authority to state, so the body may be left out.
        return await ending(runtime, runtime.stop, principal, authority or Authority())

    @robot.post("/management", status_code=202)
    def manage(runtime: Robot, principal: Caller, request: ManagementRequest) -> Operation:
        return runtime.manage(principal, request)

    @robot.get("/operations/{operation_id}")
    def operation(runtime: Robot, operation_id: str) -> Operation:
        return runtime.operation(operation_id)

    @robot.get("/events")
    async def events(
        runtime: Robot, request: Request, after: Annotated[int, Query(ge=0)] = 0, follow: bool = True
    ) -> StreamingResponse:
        """Numbered Server-Sent Events. With follow=false the retained events are sent and the stream ends."""
        if service.subscribers >= service.limits.max_event_subscribers:
            raise RobotError(
                "too_many_subscribers",
                "the event subscriber limit is reached; try again later",
                maximum=service.limits.max_event_subscribers,
            )
        # Counted here, on the event loop, where no other request runs between the check and the count.
        service.subscribers += 1

        async def feed() -> AsyncIterator[str]:
            try:
                last = after
                while True:
                    # Read off the event loop: the read waits for the runtime's lock, and the loop is what
                    # hands every stop to its workers.
                    newer, gap = await asyncio.to_thread(runtime.events, last)
                    if gap:
                        # The reader fell behind retention, or its cursor is from before a restart. Say so
                        # instead of silently skipping, and number from the events that follow.
                        yield f"event: dropped\ndata: {json.dumps({'after': last})}\n\n"
                        last = 0
                    for event in newer:
                        yield f"id: {event.sequence}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"
                        last = event.sequence
                    if not follow or service.closing.is_set() or await request.is_disconnected():
                        return
                    await asyncio.sleep(0.1)
            finally:
                service.subscribers -= 1

        # Event numbers belong to one boot; the header lets a reconnecting reader see that it changed.
        headers = {"Cache-Control": "no-store", "Robot-Boot-Id": runtime.boot_id}
        return StreamingResponse(feed(), media_type="text/event-stream", headers=headers)

    app.include_router(robot)
    return app


def _invalid_request(errors: list[dict[str, Any]]) -> RobotError:
    """Turn the validation library's report into one stable error a person can act on."""
    for error in errors:
        if error["type"] == "union_tag_invalid" and "profile" in error.get("ctx", {}).get("discriminator", ""):
            context = error["ctx"]
            return RobotError(
                "unsupported_profile",
                f"{context['tag']!r} is not a motion profile this service implements; it knows "
                f"{context['expected_tags']}",
                profile=context["tag"],
            )
        if error["type"] == "json_invalid":
            reason, position = error.get("ctx", {}).get("error", error["msg"]), error["loc"][-1]
            return RobotError("invalid_request", f"the request body is not JSON: {reason} (character {position})")
    problems = [
        {"field": ".".join(str(part) for part in error["loc"][1:]) or str(error["loc"][0]), "problem": error["msg"]}
        for error in errors
    ]
    summary = "; ".join(f"{p['field']}: {p['problem']}" for p in problems[:3])
    if len(problems) > 3:
        summary += f"; and {len(problems) - 3} more"
    return RobotError("invalid_request", summary, problems=problems)


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------


def load_backends(backend: str, options: dict[str, str]) -> dict[str, Backend]:
    """The robots a service offers: robot IDs mapped to backends, from a built-in name or `MODULE:FUNCTION`.

    The function is called with the --backend-arg options as keyword arguments, all strings, and returns the
    mapping. Its module loads only now, so a backend's dependencies are needed only when it is chosen.
    """
    path = BUILT_IN.get(backend, backend)
    module_name, _, function = path.partition(":")
    if not module_name or not function:
        raise StartupError(
            f"--backend {backend!r} is neither {', '.join(BUILT_IN)} nor MODULE:FUNCTION, such as my_robot:robots"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        hint = (
            NEEDS_MUJOCO
            if backend == "mujoco"
            else "the module must be importable here: install it, or run from its directory with PYTHONPATH=."
        )
        raise StartupError(f"cannot load the {backend} backend: {error}\n{hint}") from None
    factory = getattr(module, function, None)
    if not callable(factory):
        raise StartupError(f"{module_name} has no function named {function!r}")
    signature = inspect.signature(factory)
    try:
        signature.bind(**options)
    except TypeError as error:
        accepted = ", ".join(signature.parameters) or "none"
        raise StartupError(f"the {backend} backend {error} (its options: {accepted})") from None
    robots = factory(**options)
    if not isinstance(robots, dict) or not all(isinstance(robot, Backend) for robot in robots.values()):
        raise StartupError(f"{path} must return a dict of robot IDs to Backend instances")
    return robots


class HomeInUse(Exception):
    """serve() was given a home that another service is already running in, or starting in."""


def _load_tokens(home: Path) -> dict[str, str]:
    """Reuse a home's tokens across restarts, so a restart shows up as a stale context and not as a login failure."""
    path = home / "tokens.json"
    if path.exists():
        try:
            stored = json.loads(path.read_text())
            client, operator = stored["client"], stored["operator"]
            usable = isinstance(client, str) and isinstance(operator, str) and client and operator
            if not usable or client == operator:
                raise ValueError
        except ValueError, KeyError, TypeError:
            raise StartupError(
                f"cannot use {path}: it does not hold a client token and a different operator token; "
                "remove it to have new tokens written (clients then connect again)"
            ) from None
        return {"client": client, "operator": operator}
    tokens = {"client": secrets.token_urlsafe(24), "operator": secrets.token_urlsafe(24)}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(tokens, file, indent=2)
    return tokens


def serve(
    backend: str,
    backend_options: dict[str, str],
    home: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    log_level: str = "warning",
) -> None:
    """Run a service in this process until it is interrupted or terminated.

    `home` holds the request journal, the access tokens, and service.json, which
    tells local clients where the running service listens. A home has one service
    at a time: a second one gets HomeInUse before it changes anything there.
    """
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Termination at any point, startup included, runs the cleanup below instead of leaving files or a
    # backend's child process behind. (uvicorn shuts down gracefully on SIGTERM, then re-raises it.)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    limits = ServiceLimits()
    boot_id = "boot-" + secrets.token_hex(8)
    record = home / "service.json"
    home.mkdir(parents=True, exist_ok=True)
    # Holding the journal is what makes this service the home's only one, so it is taken before anything
    # else: every later failure then belongs to a service that did own the home.
    try:
        journal = Journal(home / "journal.sqlite", boot_id, quota=limits.journal_quota)
    except JournalInUse:
        try:
            running = json.loads(record.read_text())
            owner = f"the service at {running['url']} (pid {running['pid']})"
        except OSError, ValueError, KeyError:  # no record: its owner is still starting, or is not a service
            owner = "another process"
        raise HomeInUse(f"service home {home} is in use by {owner}") from None
    # Each of these is released when its block ends, the last one built first, however the block ends: an error
    # while starting, an interrupt, or termination.
    with journal, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener, contextlib.ExitStack() as robots:
        record.unlink(missing_ok=True)  # left by a service that was killed; this one writes its own when ready
        # Bind before the tokens are written or a backend starts, so an unusable address fails early.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, port))
        except OSError as error:
            raise StartupError(f"cannot listen on {host}:{port}: {error.strerror}") from None
        tokens = _load_tokens(home)
        principals = {
            tokens["client"]: Principal("client"),
            tokens["operator"]: Principal("operator", operator=True),
        }
        listener.listen()  # connections queue from here on, so a client may connect as soon as service.json exists
        if host not in ("127.0.0.1", "localhost"):
            log.warning("listening on %s without TLS: tokens and commands cross the network in clear text", host)
        url = f"http://{host}:{listener.getsockname()[1]}"
        backends = load_backends(backend, backend_options)
        for robot in backends.values():
            robots.enter_context(robot)
        runtimes = {
            rid: Runtime(rid, robot, journal, boot_id=boot_id, limits=limits) for rid, robot in backends.items()
        }
        description = {
            "url": url,
            "pid": os.getpid(),
            "boot_id": boot_id,
            "backend": backend,
            "robots": sorted(runtimes),
            "version": __version__,
        }
        with Service(runtimes, principals, boot_id, limits) as service:
            server = uvicorn.Server(
                uvicorn.Config(create_app(service), log_level=log_level, access_log=False, timeout_graceful_shutdown=3)
            )
            with _announced(record, description):
                print(f"homunculus {__version__} serving {', '.join(sorted(runtimes))} at {url}", flush=True)
                print(f"service home: {home}   (connect with: homunculus connect --home {home})", flush=True)
                server.run(sockets=[listener])


@contextlib.contextmanager
def _announced(record: Path, description: dict[str, Any]) -> Iterator[None]:
    """service.json, which tells local clients where the running service listens: there for the block."""
    scratch = record.with_name(record.name + ".tmp")  # renamed into place, so a reader never sees half a record
    scratch.write_text(json.dumps(description, indent=2))
    scratch.replace(record)
    try:
        yield
    finally:
        record.unlink(missing_ok=True)
