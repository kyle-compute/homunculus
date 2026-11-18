"""Adapter for one downloaded benchmark task, driven through the task's own simulator.

The task's source, physics, assets, and verifier stay where they were
downloaded; nothing is copied or modified. Either this module starts the task's
unchanged simulator entry point with an interpreter the user supplies, or it
attaches to that simulator already running, as the task's own containers run
it. It maps five profiles onto the motor controls the simulator offers over HTTP:

    base.planar_velocity/1 (timed goal)   ->  repeated `step` actions with base_velocity,
                                              judged against the base velocity the task measures;
                                              with hands_follow_base, each step also carries both
                                              hand targets with the measured base, and so does the
                                              fallback when such a goal fails or is stopped
    gripper.width/1 on the left gripper   ->  one `grip` action
    gripper.grasp/1 on the left gripper   ->  one `grip` action to closed, judged on the gap the jaws stop at
    tool.pose.linear/1 on either hand     ->  repeated `step` actions with a hand target moving along
                                              the straight segment, judged against the measured tool site
    tool.pose.linear/1 on both_hands      ->  the same for the left tool site, with the right hand target
                                              carried along rigidly in the same steps (an object held in
                                              both hands moves without being pulled between them)
    hand.posture/1 on the right hand      ->  one `grip` action to the opening the posture names

The task's world is stepped: physics advances only while an action runs. The
simulator records every physics step in its state directory, and that record
is what the task's verifier replays.

Written against `g1-dining-salt-mill`, and run with tasks whose simulators offer
the same controls and robot state (`dining-loaded-tray`, `office-paper-injection`,
whose simulator states no budgets). The limits and state fields below are those
tasks'; another task works only if its simulator offers the same controls.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import queue
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Any

import httpx

from ..grasp import GraspJudge, GripperGraspGoal, GripperGraspLimits
from ..models import (
    BackendInfo,
    BaseVelocityGoal,
    BaseVelocityLimits,
    CameraImage,
    ClockInfo,
    GripperWidthGoal,
    GripperWidthLimits,
    HandPostureGoal,
    HandPostureLimits,
    Pose,
    ProfileBinding,
    RobotError,
    Sample,
    SensorInfo,
    ToolPoseGoal,
    ToolPoseLimits,
)
from ..profiles import ActivityStatus, PoseJudge, PostureJudge, VelocityJudge, WidthJudge
from ..runtime import Backend, BackendFacts, Feedback, Frame, GoalCommand, ResourceSpec, StartupError

log = logging.getLogger(__name__)

CAMERAS = ("head", "eyes", "left_wrist", "right_wrist")
# Bounds the task's own `step` and `grip` actions enforce, restated here for admission.
STEP_MIN_S, STEP_MAX_S = 0.02, 0.25
MAX_GAP_M = 0.10
# Standstill is measured, not assumed: the task must report a stable stand (both feet loaded, upright,
# slow), or a gap that changes less than GAP_STILL_M per step, for SETTLE_S within SETTLE_LIMIT_S.
SETTLE_S, SETTLE_STEP_S, SETTLE_LIMIT_S = 0.2, 0.05, 4.0
GAP_STILL_M = 0.0002
# The arms and right fingers count as at rest when they move less than this per settle step. The arm's default
# suits g1-dining-salt-mill's stiff arms; softer arms holding a load tremble more (the arm_still_m option).
TOOL_STILL_M, FINGER_STILL_RAD = 0.0005, 0.002
# A pose goal sends the moving hand target this often; the task's own scripted moves use 0.1 s steps.
POSE_STEP_S = 0.1
SIDES = ("left", "right")
ARMS = tuple(f"{side}_arm" for side in SIDES)
ARM_JOINTS = {
    side: [
        f"{side}_{part}_joint"
        for part in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
    for side in SIDES
}
# The right hand's finger angles for each posture. "open" is the task's finger targets at opening 1 (humanoid.py
# finger_targets), reached within 0.016 rad by a hand clear of the body. "closed" is measured, not the task's
# opening-0 targets: closing an empty hand stops on finger-to-finger contact (38-40 N) about 0.27 rad short of
# them. These are its angles over three closures on Modal, 2026-11-16, which agreed within 0.027 rad.
RIGHT_POSTURES = {
    "open": {
        "thumb_0": -0.55,
        "thumb_1": -0.15,
        "thumb_2": 0.0,
        "middle_0": 0.0,
        "middle_1": 0.0,
        "index_0": 0.0,
        "index_1": 0.0,
    },
    "closed": {
        "thumb_0": -0.60,
        "thumb_1": -0.72,
        "thumb_2": -1.24,
        "middle_0": 1.095,
        "middle_1": 1.449,
        "index_0": 0.85,
        "index_1": 1.40,
    },
    # Partly open: where an empty hand rests at openings 0.5, 0.4 and 0.3 (local macOS run, 2026-11-17), within
    # 0.02 rad of the task's own targets. Below 0.3 an empty hand's fingers already touch each other, so no
    # posture is offered between open_30 and closed. Closed on an object, these press more gently than closed.
    "open_50": {
        "thumb_0": -0.548,
        "thumb_1": -0.56,
        "thumb_2": -0.608,
        "middle_0": 0.531,
        "middle_1": 0.708,
        "index_0": 0.531,
        "index_1": 0.708,
    },
    "open_40": {
        "thumb_0": -0.548,
        "thumb_1": -0.644,
        "thumb_2": -0.733,
        "middle_0": 0.64,
        "middle_1": 0.852,
        "index_0": 0.64,
        "index_1": 0.852,
    },
    "open_30": {
        "thumb_0": -0.547,
        "thumb_1": -0.729,
        "thumb_2": -0.858,
        "middle_0": 0.75,
        "middle_1": 0.997,
        "index_0": 0.75,
        "index_1": 0.997,
    },
}
POSTURE_OPENING = {"open": 1.0, "open_50": 0.5, "open_40": 0.4, "open_30": 0.3, "closed": 0.0}
# The pose target that moves both hands: the left tool site is commanded and judged; the right hand target keeps
# its pose relative to it. The right hand must stay within the path tolerance of where it is carried.
BOTH_HANDS = "both_hands"
# Hand pose goals. The workspace and speeds are the bounds the task's own hand controls enforce. The
# tolerances were checked against 48 free-space goals on Modal: see docs/guide.md.
POSE_LIMITS = {
    "profile": "tool.pose.linear/1",
    "frames": ["world"],
    "workspace_min_m": [-2.85, -2.85, 0.55],
    "workspace_max_m": [2.85, 2.85, 1.35],
    "min_speed_m_s": 0.01,
    "max_speed_m_s": 0.20,
    "default_speed_m_s": 0.08,
    "max_rotation_rate_rad_s": 0.5,
    "path_tolerance_m": 0.05,
    "position_tolerance_m": 0.01,
    "max_position_tolerance_m": 0.05,
    "orientation_tolerance_rad": 0.08,
    "max_orientation_tolerance_rad": 0.3,
    "settle_time_s": SETTLE_S,
    "max_settle_wait_s": 3.0,
}
# How base travel is verified, qualified on recorded free and blocked walks of this task's walking policy.
# Velocity is averaged over one 0.75 s gait cycle; anything
# shorter mostly measures sway. Over that window the policy delivered 40-125 % of each commanded component
# when free and at most 22 % against an obstacle, and it stands instead of walking below 0.05.
TRACKING = {
    "transient_s": 1.0,
    "max_tracking_error_m_s": 0.05,
    "max_tracking_error_rad_s": 0.12,
    "max_tracking_error_ratio": 0.7,
    "max_tracking_error_duration_s": 0.75,
    "tracking_window_s": 0.75,
}
ZERO_VELOCITY = {"forward_m_s": 0.0, "left_m_s": 0.0, "yaw_rate_rad_s": 0.0}
FOLLOW_NOTE = (
    "; while it walks, both hand targets move and turn with the measured base, so the hands keep their pose "
    "relative to it, as when carrying something in both hands, and the goal ends once the arms are at rest too; "
    "if the goal fails or is stopped, the hands keep those targets, not the poses they are measured at"
)
# Passed through to the simulator when set: the task's own settings, and what its libraries need.
SIMULATOR_ENV = ("SCENE_SEED", "POSE_JITTER", "RENDER_BACKEND", "MUJOCO_GL", "PATH", "TMPDIR")
# Runs the entry point as a script, with its own directory first on the import path. In the task's container that
# directory is /opt/simulator, and an entry point may put that path, not its own, on the import path (isolated mode
# leaves the script's directory off it); here the simulator sits in the downloaded task instead.
ENTRY_POINT = (
    "import os, runpy, sys; entry = sys.argv[1]; sys.argv = [entry]; "
    "sys.path.insert(0, os.path.dirname(entry)); runpy.run_path(entry, run_name='__main__')"
)


def robots(
    task_dir: str | None = None,
    python: str | None = None,
    state_dir: str | None = None,
    url: str | None = None,
    name: str | None = None,
    hands_follow_base: str = "false",
    arm_still_m: str = str(TOOL_STILL_M),
) -> dict[str, Backend]:
    """The robot `--backend task` serves.

    To start the simulator: `task_dir`, the downloaded task; `python`, an interpreter with the task's own
    requirements; `state_dir`, a new directory for the recorded episode. Paths are made absolute but keep
    their symlinks: a virtual environment's interpreter is a symlink, and resolving it would select the base
    interpreter.

    To attach to a simulator that is already running, such as the task's own simulator container: `url`, where
    it listens, and `name`, the task's name. That simulator records the episode in its own state directory.

    `hands_follow_base=true`: a base goal also carries both hand targets with the measured base, so the hands keep
    their pose relative to it while it walks, as when carrying something in both hands. Otherwise they stay where
    they are in the world. When such a goal fails or is stopped, the arms keep their carried targets (still
    carried with the base while it comes to rest) instead of holding where they are measured: they were holding
    those targets, not reaching for them, and a loaded arm measured where it sags would let its load sink and turn
    with every failed step.

    `arm_still_m`: how far a hand's tool site may move in one 0.05 s step while its arm counts as at rest, which
    confirms that a stopped arm holds still. Arms whose controller trembles under a load need more than the
    default, or no stop of a loaded arm is ever confirmed.
    """
    if hands_follow_base not in ("true", "false"):
        raise StartupError(f"hands_follow_base is true or false, not {hands_follow_base!r}")
    try:
        still = float(arm_still_m)
    except ValueError:
        still = math.nan
    if not 0 < still < 0.05:
        raise StartupError(f"arm_still_m is a distance in metres between 0 and 0.05, not {arm_still_m!r}")
    options = {"hands_follow_base": hands_follow_base == "true", "arm_still_m": still}
    if url is not None:
        if task_dir or python or state_dir or not name:
            raise StartupError(
                "give url and name to attach to a running simulator, or task_dir, python and state_dir to start one"
            )
        task = TaskBackend(name, url.rstrip("/"), epoch=url.rstrip("/"), **options)
        return {f"task-{task.name}": task}
    if not (task_dir and python and state_dir) or name:
        raise StartupError(
            "give task_dir, python and state_dir to start the simulator, or url and name to attach to one"
        )
    task_path, python_path, state_path = (
        Path(os.path.abspath(Path(v).expanduser())) for v in (task_dir, python, state_dir)
    )
    task = TaskBackend(
        task_path.name,
        epoch=str(state_path),
        state_dir=state_path,
        simulator=_start_simulator(task_path, python_path, state_path),
        **options,
    )
    return {f"task-{task.name}": task}


@contextmanager
def _start_simulator(task_dir: Path, python: Path, state_dir: Path) -> Iterator[tuple[subprocess.Popen[bytes], str]]:
    """Start the task's own simulator entry point recording into state_dir; return it and its URL."""
    entry = task_dir / "environment" / "simulator" / "start.py"
    if not entry.is_file():
        raise StartupError(f"{task_dir} is not a task with a simulator: {entry} does not exist")
    if not python.is_file():
        raise StartupError(
            f"the task interpreter {python} (option python) does not exist; docs/guide.md shows how to create it"
        )
    if (state_dir / "episode.json").exists() or (state_dir / "trajectory").exists():
        raise StartupError(f"{state_dir} already holds a recording; choose a new state_dir")
    state_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # The same entry point the task's own container runs, isolated from this environment and
    # told not to write bytecode into the downloaded tree. It records into state_dir only.
    # Downloaded code gets the settings it reads, not the whole environment of this service.
    settings = {name: os.environ[name] for name in SIMULATOR_ENV if name in os.environ}
    with (
        open(state_dir / "simulator.log", "w") as output,
        subprocess.Popen(
            [str(python), "-I", "-B", "-c", ENTRY_POINT, str(entry)],
            cwd=state_dir,
            env={**settings, "STATE_DIR": str(state_dir), "SIM_PORT": str(port)},
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        ) as process,
    ):
        try:
            yield process, f"http://127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class _Refused(Exception):
    """The task simulator rejected an action (finished episode, exhausted budget, fallen robot)."""


class _Cancelled(Exception):
    """The worker stopped sending steps for an overtaken activity."""


class TaskBackend(Backend):
    reports_goal_samples = True

    def __init__(
        self,
        name: str,
        url: str = "",
        *,
        epoch: str,
        state_dir: Path | None = None,
        simulator: AbstractContextManager[tuple[subprocess.Popen[bytes], str]] | None = None,
        hands_follow_base: bool = False,
        arm_still_m: float = TOOL_STILL_M,
    ):
        """Start the owned simulator or attach to the given URL."""
        self.name = name
        self._arm_still_m = arm_still_m
        self._follows = hands_follow_base  # a base goal then also commands both arms (see robots())
        self._base_resources = ["base", *ARMS] if hands_follow_base else ["base"]
        self._epoch = epoch  # names this episode on the task_sim clock
        self._state_dir = state_dir
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()  # guards everything below, shared by the worker and the control thread
        self._robot: dict[str, Any] = {}
        self._sim_time = 0.0
        self._task: dict[str, Any] = {}
        self._finished = False
        self._reachable = False  # true while the simulator's latest answer was a successful one
        self._activities: dict[str, ActivityStatus] = {}
        self._owner: dict[str, str] = {}  # resource id -> activity commanding it
        # Base goals that carry the hand targets (hands_follow_base): activity id -> the carrier its steps use.
        self._carriers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self._cancelled: set[str] = set()
        self._closing = threading.Event()
        self._commands: queue.Queue[Callable[[], None] | None] = queue.Queue()
        with ExitStack() as resources:
            if simulator is not None:
                self._process, url = resources.enter_context(simulator)
            self._http = resources.enter_context(httpx.Client(base_url=url, timeout=300.0, trust_env=False))
            self._read_initial_state(url)
            self._worker = threading.Thread(target=self._work, name=f"task:{self.name}")
            resources.callback(self._stop_worker)
            self._worker.start()
            self._resources = resources.pop_all()

    def _read_initial_state(self, url: str) -> None:
        try:
            self._wait_until_ready()
            self._record(self._http.get("/state").raise_for_status().json())
            # The capture budget, when the simulator states one (in /state, as g1-dining-salt-mill does).
            self._captures: int | None = self._task.get("observations_remaining")
            self._revisions = {"task": self.name, "simulator": url}
            if self._state_dir is not None:
                self._revisions["scene_seed"] = str(json.loads((self._state_dir / "seed.json").read_text())["seed"])
                self._revisions["state_dir"] = str(self._state_dir)
            hands = self._robot["hands"]
            # What each hand's controller holds while no goal commands it: where it was measured at start.
            self._hand_target = {side: _tool_pose(hands[side]) for side in SIDES}
            # Per resource: measured at rest under the fallback. The world is stepped, so before the first
            # action nothing is in motion; the base still has to be measured standing.
            self._settled = {
                "base": bool(self._robot["support"]["stable"]),
                "left_gripper": True,
                "left_arm": True,
                "right_arm": True,
                "right_hand": True,
            }
            self.advance()  # fails here, not later, if this task reports a different robot state
        except (KeyError, TypeError, ValueError, httpx.HTTPError) as error:
            raise StartupError(
                f"the simulator of {self.name} does not report the robot state this adapter maps "
                f"({type(error).__name__}: {error}); it was written for g1-dining-salt-mill"
            ) from None

    def _alive(self) -> bool:
        """Whether the simulator can still answer: a started process has not exited (an attached one is assumed
        up, and reachability shows otherwise)."""
        return self._process is None or self._process.poll() is None

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 180.0
        while True:
            if not self._alive() or time.monotonic() > deadline:
                if self._state_dir is None:
                    raise StartupError(f"the task simulator at {self._http.base_url} did not answer /health in time")
                tail = (self._state_dir / "simulator.log").read_text(errors="replace")[-1500:]
                raise StartupError(f"the task simulator did not start; its log ends with:\n{tail}")
            try:
                if self._http.get("/health", timeout=2.0).json().get("ready"):
                    return
            except httpx.TransportError, ValueError:
                pass
            time.sleep(0.2)

    # -- what the runtime asks --------------------------------------------

    def describe(self) -> BackendFacts:
        with self._lock:
            finished, alive = self._finished, self._alive()
        reason = None
        if finished:
            reason = "the episode is finished; a new episode needs a new simulator and a new service"
        elif not alive:
            reason = "the task simulator process is not running"
        common = {"modes": ["goal"], "available": reason is None, "unavailable_reason": reason}
        return BackendFacts(
            info=BackendInfo(
                kind="task",
                physics="simulated",
                notice=(
                    f"Downloaded task {self.name!r} running its own unchanged MuJoCo simulator. "
                    "Results are simulated physics, not a physical robot."
                ),
            ),
            world_id=f"{self.name}:{self._epoch}",
            mode="task_episode",
            clock_mode="stepped",
            clocks=[
                ClockInfo(
                    id="task_sim",
                    epoch=self._epoch,
                    description="the task simulator's time since episode start; it advances only while an action runs",
                )
            ],
            frames=["world", "base_heading"],
            sensors=[
                SensorInfo(
                    id="base",
                    kind="base_state",
                    provenance="simulation_ground_truth",
                    description="base pose [x, y, yaw], pelvis height, foot support, and balance state from the task",
                ),
                SensorInfo(
                    id="left_gripper",
                    kind="joint_state",
                    provenance="simulation_ground_truth",
                    description="measured gap of the left parallel gripper and its contact force",
                ),
                *(
                    SensorInfo(
                        id=f"{side}_arm",
                        kind="joint_state",
                        provenance="simulation_ground_truth",
                        description=f"{side} arm joint angles and the measured pose of its {side}_tool site",
                    )
                    for side in SIDES
                ),
                SensorInfo(
                    id="right_hand",
                    kind="joint_state",
                    provenance="simulation_ground_truth",
                    description="right hand finger joint angles and their contact force",
                ),
                SensorInfo(
                    id="task",
                    kind="task_state",
                    provenance="simulation_ground_truth",
                    description="hand poses, remaining budgets, and whether the episode is finished",
                ),
                *(
                    SensorInfo(
                        id=camera,
                        kind="camera",
                        provenance="simulation_ground_truth",
                        description="rendered 960x720 RGB and metric depth"
                        + (
                            f"; each capture uses the task's budget of {self._captures}"
                            if self._captures is not None
                            else ""
                        ),
                    )
                    for camera in CAMERAS
                ),
            ],
            revisions=dict(self._revisions),
            resources=[
                ResourceSpec(id="base", role="mobile_base", joints=[], controller="task locomotion policy"),
                ResourceSpec(
                    id="left_gripper",
                    role="left_gripper",
                    joints=["left_jaw_a", "left_jaw_b"],
                    controller="task gripper controller",
                ),
                *(
                    ResourceSpec(
                        id=f"{side}_arm", role=f"{side}_arm", joints=ARM_JOINTS[side], controller="task arm controller"
                    )
                    for side in SIDES
                ),
                ResourceSpec(
                    id="right_hand",
                    role="right_hand",
                    joints=[f"right_hand_{part}_joint" for part in RIGHT_POSTURES["open"]],
                    controller="task finger controller",
                ),
            ],
            bindings=[
                ProfileBinding(
                    profile="base.planar_velocity/1",
                    target="base",
                    role="mobile_base",
                    resources=list(self._base_resources),
                    controller="task locomotion policy",
                    assistance=(
                        "the task's learned walking policy keeps balance and follows the velocity loosely; travel is "
                        "verified against base velocity measured over one gait cycle, so a blocked base fails"
                        + (FOLLOW_NOTE if self._follows else "")
                    ),
                    stop_deadline_wall_s=15.0,
                    limits=BaseVelocityLimits(
                        profile="base.planar_velocity/1",
                        control_point="pelvis",  # the body whose position and heading the task reports as the base
                        frames=["base_heading"],
                        max_forward_m_s=0.35,
                        max_left_m_s=0.25,
                        max_yaw_rate_rad_s=0.5,
                        max_goal_duration_s=10.0,
                        **TRACKING,
                        stop_speed_m_s=0.08,
                        stop_yaw_rate_rad_s=0.25,
                        settle_time_s=SETTLE_S,
                    ),
                    **common,
                ),
                ProfileBinding(
                    profile="gripper.width/1",
                    target="left_gripper",
                    role="left_gripper",
                    resources=["left_gripper"],
                    controller="task gripper controller",
                    assistance="the task ramps the jaw opening over one simulated second; grasping is by contact only",
                    stop_deadline_wall_s=15.0,
                    limits=GripperWidthLimits(
                        profile="gripper.width/1",
                        min_width_m=0.0,
                        max_width_m=MAX_GAP_M,
                        speed_m_s=MAX_GAP_M,
                        goal_tolerance_m=0.003,
                        settle_time_s=SETTLE_S,
                        # The task reports a contact force, but no qualified contact rule exists for it.
                        complete_on=["width_reached"],
                    ),
                    **common,
                ),
                ProfileBinding(
                    profile="gripper.grasp/1",
                    target="left_gripper",
                    role="left_gripper",
                    resources=["left_gripper"],
                    controller="task gripper controller",
                    assistance="the task closes the jaws over one simulated second; they keep squeezing afterwards",
                    stop_deadline_wall_s=15.0,
                    limits=GripperGraspLimits(
                        profile="gripper.grasp/1",
                        max_width_m=MAX_GAP_M,
                        empty_width_m=0.01,  # closed on nothing, the jaws measure about 0
                        still_tolerance_m=GAP_STILL_M,
                        settle_time_s=SETTLE_S,
                    ),
                    **common,
                ),
                *(
                    ProfileBinding(
                        profile="tool.pose.linear/1",
                        target=f"{side}_hand",
                        role=f"{side}_arm",
                        resources=[f"{side}_arm"],
                        controller="task arm controller",
                        assistance=(
                            "the task's whole-arm inverse kinematics follows the moving hand target and avoids "
                            "self-collision; it does not plan around the scene, and the hand target "
                            + (
                                "moves with the base when it walks"
                                if self._follows
                                else "stays fixed in the world when the base walks"
                            )
                        ),
                        stop_deadline_wall_s=15.0,
                        limits=ToolPoseLimits(**POSE_LIMITS, control_point=f"{side}_tool"),
                        **common,
                    )
                    for side in SIDES
                ),
                ProfileBinding(
                    profile="tool.pose.linear/1",
                    target=BOTH_HANDS,
                    role="both_arms",
                    resources=["left_arm", "right_arm"],
                    controller="task arm controller",
                    assistance=(
                        "moves the left tool site like a left_hand goal and, in the same steps, carries the right "
                        "hand target with it rigidly, so the right hand keeps its pose relative to the left; for an "
                        "object held in both hands. The goal also fails if the right hand is measured more than the "
                        "path tolerance from where it is carried"
                    ),
                    stop_deadline_wall_s=15.0,
                    limits=ToolPoseLimits(**POSE_LIMITS, control_point="left_tool"),
                    **common,
                ),
                ProfileBinding(
                    profile="hand.posture/1",
                    target="right_hand",
                    role="right_hand",
                    resources=["right_hand"],
                    controller="task finger controller",
                    assistance=(
                        "the task ramps the fingers over one simulated second; closed is where an empty hand's "
                        "fingers stop on each other, and open_50, open_40 and open_30 are partly open (50, 40 and "
                        "30 percent); grasping is by contact only, so a hand closed on an object stops short of it, "
                        "and a more open posture presses on it more gently"
                    ),
                    stop_deadline_wall_s=15.0,
                    limits=HandPostureLimits(
                        profile="hand.posture/1",
                        postures={
                            name: {f"right_hand_{part}_joint": angle for part, angle in joints.items()}
                            for name, joints in RIGHT_POSTURES.items()
                        },
                        max_speed_rad_s=1.5,
                        goal_tolerance_rad=0.06,
                        settle_time_s=SETTLE_S,
                    ),
                    **common,
                ),
            ],
            # The task could move several in one step; this adapter does not, so no combination is offered.
            concurrent=[],
            management=["recover"] if finished else ["episode.finish", "recover"],
            period_s=0.05,
        )

    def apply(self, activity_id: str, contract: GoalCommand) -> None:
        run: Callable[..., None] = {
            BaseVelocityGoal: self._travel,
            GripperWidthGoal: self._grip,
            GripperGraspGoal: self._grasp,
            ToolPoseGoal: self._pose,
            HandPostureGoal: self._posture,
        }[type(contract.action)]
        with self._lock:
            self._activities[activity_id] = ActivityStatus("executing")
            for rid in contract.resources:
                self._owner[rid] = activity_id
                self._settled[rid] = False
        self._commands.put(lambda: run(activity_id, contract.judge))

    def stop(self, resources: Sequence[str]) -> None:
        with self._lock:
            # Stopping one resource of a goal that owns several stops the goal, so all of them come to rest.
            stopped = {self._owner[rid] for rid in resources if rid in self._owner}
            resources = list(dict.fromkeys([*resources, *(r for r, a in self._owner.items() if a in stopped)]))
            # A stopped base goal that carried the hand targets keeps carrying them while the base comes to rest.
            carry = next((self._carriers.pop(a) for a in stopped if a in self._carriers), None)
            for rid in resources:
                activity_id = self._owner.pop(rid, None)
                if activity_id is not None:
                    self._cancelled.add(activity_id)
                    self._activities.pop(activity_id, None)
                if not self._finished:
                    self._settled[rid] = False  # until the fallback queued below measures rest
            if self._finished:
                return  # a finished episode never steps again: what was measured before it ended stands
        self._commands.put(lambda: self._fallback(list(resources), carry=carry))

    def manage(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any] | Future[dict[str, Any]]:
        if operation == "recover":
            self.stop(parameters["resources"])
            return {}
        if operation != "episode.finish":
            raise RobotError("unsupported_operation", f"the task adapter has no {operation}", operation=operation)
        # Ends the episode the way the task defines it. Queued behind any action still running.
        result: Future[dict[str, Any]] = Future()

        def confirmed() -> bool:
            with self._lock:
                if self._finished:
                    result.set_result({**self._task, "finished": True, "sim_time_s": self._sim_time})
                    return True
            return False

        def confirm() -> None:
            if self._closing.is_set():
                return
            try:
                reply = self._http.get("/state", timeout=2.0).raise_for_status().json()
                with self._lock:
                    self._record(reply)
                if confirmed():
                    return
            except httpx.HTTPError, KeyError, TypeError, ValueError:
                log.debug("could not confirm episode finish", exc_info=True)
            # Queue another reading behind any stop that arrived while this request was in flight.
            if not self._closing.wait(0.2):
                self._commands.put(confirm)

        def finish() -> None:
            if result.cancelled() or self._closing.is_set():
                result.cancel()
                return
            try:
                with self._lock:
                    moving = [rid for rid, settled in self._settled.items() if not settled]
                if moving:
                    # A paused world proves nothing about the motion in it. What was cut short is brought to
                    # rest, and measured at rest, while the episode can still step.
                    self._fallback(moving)
            except Exception as error:
                if result.set_running_or_notify_cancel():
                    result.set_exception(error)
                return
            if self._closing.is_set():
                result.cancel()
            if not result.set_running_or_notify_cancel():
                return
            try:
                self._act({"action": "finish"}, timeout=60.0)
            except httpx.TransportError, ValueError, KeyError, TypeError:
                log.warning("the simulator reply to episode.finish was lost and will be checked against /state")
                with self._lock:
                    self._reachable = False
                self._commands.put(confirm)
            except Exception as error:
                result.set_exception(error)
            else:
                if not confirmed():
                    self._commands.put(confirm)

        with self._lock:
            self._cancelled.update(self._owner.values())
            self._owner.clear()
            self._activities.clear()
            self._carriers.clear()
        self._commands.put(finish)
        return result

    def capture(self, sensor: str) -> list[Frame]:
        try:
            response = self._http.get("/observe", params={"camera": sensor})
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RobotError("backend_error", f"the task simulator did not answer the capture: {error}") from error
        if response.status_code != 200:
            raise RobotError("backend_error", f"the task simulator refused the capture: {body.get('error')}")
        with self._lock:  # a simulator that counts captures states the count only in /state; each capture spends one
            if "observations_remaining" in self._task:
                self._task["observations_remaining"] -= 1
        rgb, depth = _camera_image(body["metadata"], self._epoch), base64.b64decode(body["depth_npy"])
        return [
            Frame(rgb, base64.b64decode(body["rgb_png"])),
            Frame(rgb.model_copy(update={"kind": "depth", "media_type": "application/x-npy"}), depth),
        ]

    def advance(self) -> Feedback:
        """Report the simulator's latest answer. Nothing is stepped here; the worker does that."""
        with self._lock:
            robot, valid = self._robot, self._reachable and self._alive()
            left, right = robot["hands"]["left"], robot["hands"]["right"]
            x, y, yaw = robot["base_xy_yaw"]
            support = robot["support"]
            groups = {
                "base": {
                    "frame_id": "world",
                    "positions": {"x_m": x, "y_m": y, "yaw_rad": yaw},
                    "velocities": {
                        "linear_speed_m_s": support["linear_speed_m_s"],
                        "angular_speed_rad_s": support["angular_speed_rad_s"],
                    },
                    "pelvis_height_m": robot["pelvis_height"],
                    "support": {"stable": support["stable"], "foot_normal_force_N": support["foot_normal_force_N"]},
                    "fallen": robot["fallen"],
                },
                "left_gripper": {
                    "positions": {"gap_m": left["gap_m"]},
                    "velocities": {},
                    "width_m": left["gap_m"],
                    "requested_width_m": left["requested_opening"] * MAX_GAP_M,
                    "touch_normal_force_N": left["touch_normal_force_N"],
                },
                **{
                    f"{side}_arm": {
                        "positions": dict(zip(ARM_JOINTS[side], robot["hands"][side]["arm_joint_angles"], strict=True)),
                        "velocities": {},
                        "tool": _tool_pose(robot["hands"][side]),
                        "controller_status": robot["hands"][side]["controller_status"],
                    }
                    for side in SIDES
                },
                "right_hand": {
                    "positions": dict(right["finger_joint_angles"]),
                    "velocities": {},
                    "requested_opening": right["requested_opening"],
                    "touch_normal_force_N": right["touch_normal_force_N"],
                },
                "task": {
                    "sim_time_s": self._sim_time,
                    "finished": self._finished,
                    "hands": {side: _tool_pose(hand) for side, hand in robot["hands"].items()},
                    **self._task,
                },
            }
            samples = {
                key: Sample(
                    source=f"task:{self.name}:{key}",
                    provenance="simulation_ground_truth",
                    clock_id="task_sim",
                    clock_epoch=self._epoch,
                    capture_time_ns=str(round(self._sim_time * 1e9)),
                    valid=valid,
                    values=values,
                )
                for key, values in groups.items()
            }
            feedback = Feedback(
                execution_time_s=self._sim_time,
                samples=samples,
                activities=dict(self._activities),
                stopped={rid: valid and settled and rid not in self._owner for rid, settled in self._settled.items()},
                holding={
                    "base": dict(ZERO_VELOCITY),
                    "left_gripper": {"gap_m": left["requested_opening"] * MAX_GAP_M},
                    **{f"{side}_arm": _flat_pose(self._hand_target[side]) for side in SIDES},
                    "right_hand": {"opening": right["requested_opening"]},
                },
            )
            for activity_id, status in list(self._activities.items()):
                if status.phase == "done":  # reported once; its resource then holds what it reached
                    del self._activities[activity_id]
                    self._carriers.pop(activity_id, None)
                    self._owner = {rid: owner for rid, owner in self._owner.items() if owner != activity_id}
            return feedback

    def close(self) -> None:
        self._resources.close()

    def _stop_worker(self) -> None:
        self._closing.set()
        self._commands.put(None)
        if self._worker.ident is not None:
            self._worker.join()

    # -- the worker: every simulator action runs here, in order ------------

    def _work(self) -> None:
        while (command := self._commands.get()) is not None:
            try:
                command()
            except _Cancelled:
                continue
            except Exception:
                # Feedback stays invalid until the simulator answers again, so nothing is confirmed meanwhile.
                log.exception("task simulator command failed")
                with self._lock:
                    self._reachable = False

    def _act(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        response = self._http.post("/act", json=request, timeout=self._http.timeout if timeout is None else timeout)
        if response.status_code == 400:
            raise _Refused(response.json().get("error", "rejected"))
        response.raise_for_status()
        reply = response.json()
        with self._lock:
            self._record(reply)
        return reply

    def _record(self, reply: dict[str, Any]) -> None:
        self._reachable = True
        self._robot = reply["robot"]
        self._sim_time = float(reply["sim_time"])
        self._finished = bool(reply.get("finished", self._finished))
        self._task.update({key: reply[key] for key in ("budget", "observations_remaining") if key in reply})

    def _skip(self, activity_id: str) -> bool:
        """True once a stop has overtaken this activity; the stop queued behind it applies the fallback."""
        with self._lock:
            cancelled = activity_id in self._cancelled or self._closing.is_set()
            self._cancelled.discard(activity_id)
            if cancelled:
                self._carriers.pop(activity_id, None)
            return cancelled

    def _report(self, activity_id: str, status: ActivityStatus) -> None:
        with self._lock:
            if activity_id in self._activities:  # a stopped activity is no longer reported
                self._activities[activity_id] = status

    def _travel(self, activity_id: str, judge: VelocityJudge) -> None:
        """Timed travel as equal `step` actions, then steps at rest. The judge decides from the measured poses.
        With hands_follow_base, every step also sends both hand targets, carried with the base as last measured."""
        action = judge.command
        count = max(1, math.ceil(action.duration_s / STEP_MAX_S))
        seconds = max(STEP_MIN_S, action.duration_s / count)
        command = [action.forward_m_s, action.left_m_s, action.yaw_rate_rad_s]
        carry = self._carrier() if self._follows else None
        if carry is not None:
            with self._lock:
                self._carriers[activity_id] = carry  # for the fallback, if this goal fails or is stopped
        try:
            for _ in range(count):
                if self._skip(activity_id):
                    return
                request: dict[str, Any] = {"action": "step", "seconds": seconds, "base_velocity": command}
                if carry is not None:
                    with self._lock:
                        robot = self._robot
                    request["hands"] = carry(robot)
                reply = self._act(request)
                if reply["robot"]["fallen"]:
                    self._report(activity_id, ActivityStatus("failed", error="robot_fell", detail="the robot fell"))
                    return
                status = judge.update(reply["sim_time"], tuple(reply["robot"]["base_xy_yaw"]), still_for_s=0.0)
                if status.phase == "failed":
                    self._report(activity_id, status)
                    return
            self._report(activity_id, ActivityStatus("finishing"))
            rested, _ = self._rest(self._base_resources, activity_id=activity_id, carry=carry)
        except _Refused as refusal:
            self._report(activity_id, ActivityStatus("failed", error="task_rejected", detail=str(refusal)))
            return
        with self._lock:
            time_s, pose, stable = self._sim_time, tuple(self._robot["base_xy_yaw"]), self._robot["support"]["stable"]
            for rid in self._base_resources:
                self._settled[rid] = rested >= SETTLE_S
        status = judge.update(time_s, pose, still_for_s=rested)  # where the base came to rest
        if status.phase != "done":  # no stable stand was measured in the time this adapter steps for
            detail = "the base did not come to a stable stand after travelling" + (
                ", with both arms at rest" if carry is not None else ""
            )
            status = ActivityStatus("failed", judge.measurements(), error="base_not_settled", detail=detail)
        status.measurements["stable"] = stable
        self._report(activity_id, status)

    def _carrier(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Hand targets for a step, from the robot state before it: each hand target where it was when the travel
        began, moved by the planar motion that takes the base from where it was measured then to where it is
        measured now. Hand targets, not measured hands, are carried, since a loaded hand sags below its target."""
        with self._lock:
            x0, y0, yaw0 = self._robot["base_xy_yaw"]
            held = {side: self._hand_target[side] for side in SIDES}

        def carry(robot: dict[str, Any]) -> dict[str, Any]:
            x, y, yaw = robot["base_xy_yaw"]
            turn = [0.0, 0.0, math.sin((yaw - yaw0) / 2), math.cos((yaw - yaw0) / 2)]
            targets = {}
            for side, pose in held.items():
                px, py, pz = pose["position_m"]
                dx, dy, _ = _rotate(turn, [px - x0, py - y0, 0.0])
                targets[side] = {
                    "frame_id": "world",
                    "position_m": [x + dx, y + dy, pz],
                    "orientation_xyzw": _qmul(turn, pose["orientation_xyzw"]),
                }
            with self._lock:
                self._hand_target.update(targets)
            return {
                side: {
                    "position": t["position_m"],
                    "quaternion": [t["orientation_xyzw"][3], *t["orientation_xyzw"][:3]],
                }
                for side, t in targets.items()
            }

        return carry

    def _grip(self, activity_id: str, judge: WidthJudge) -> None:
        """One `grip` action, then steps at rest. The judge decides from the gap measured after each step."""
        if self._skip(activity_id):
            return
        action = judge.action
        try:
            self._act({"action": "grip", "hand": "left", "opening": action.width_m / MAX_GAP_M})
            self._report(activity_id, ActivityStatus("finishing"))
            rested, status = self._rest(
                ["left_gripper"],
                activity_id=activity_id,
                until=lambda robot: judge.update(robot["hands"]["left"]["gap_m"], SETTLE_STEP_S),
            )
        except _Refused as refusal:
            self._report(activity_id, ActivityStatus("failed", error="task_rejected", detail=str(refusal)))
            return
        with self._lock:
            hand = self._robot["hands"]["left"]
            self._settled["left_gripper"] = rested >= SETTLE_S
        if status.phase != "done":  # the gap never held at the width in the time this adapter steps for
            measurements = {"width_m": hand["gap_m"], "error_m": abs(hand["gap_m"] - action.width_m)}
            detail = "the jaws did not hold the requested width; something may be between them"
            status = ActivityStatus("failed", measurements, error="width_not_reached", detail=detail)
        status.measurements.update(touch_normal_force_N=hand["touch_normal_force_N"], at_rest=rested >= SETTLE_S)
        self._report(activity_id, status)

    def _grasp(self, activity_id: str, judge: GraspJudge) -> None:
        """One `grip` action to closed, then steps at rest until the judge decides from the gap the jaws stopped at.
        The jaws stay asked to close, so whatever they hold stays squeezed after the goal ends."""
        if self._skip(activity_id):
            return
        try:
            self._act({"action": "grip", "hand": "left", "opening": 0.0})
            self._report(activity_id, ActivityStatus("finishing"))
            rested, status = self._rest(
                ["left_gripper"],
                activity_id=activity_id,
                until=lambda robot: judge.update(robot["hands"]["left"]["gap_m"], SETTLE_STEP_S),
            )
        except _Refused as refusal:
            self._report(activity_id, ActivityStatus("failed", error="task_rejected", detail=str(refusal)))
            return
        with self._lock:
            hand = self._robot["hands"]["left"]
            self._settled["left_gripper"] = rested >= SETTLE_S
        if status.phase not in ("done", "failed"):
            detail = f"the jaws did not stop within {SETTLE_LIMIT_S:g} simulated seconds"
            status = ActivityStatus("failed", {"width_m": hand["gap_m"]}, error="grasp_not_settled", detail=detail)
        status.measurements.update(touch_normal_force_N=hand["touch_normal_force_N"])
        self._report(activity_id, status)

    def _pose(self, activity_id: str, judge: PoseJudge) -> None:
        """Steps that move the hand target along the judge's reference, then steps holding the goal pose, until
        the judge decides from the tool site measured after each step.

        On both_hands the left tool site is judged, and each step also sends the right hand target, carried
        along by the same rigid motion that takes the reference from its start to where it is now."""
        action, limits = judge.action, judge.limits
        both = action.target == BOTH_HANDS
        side = "left" if both else action.target.removesuffix("_hand")
        rids = ["left_arm", "right_arm"] if both else [f"{side}_arm"]
        with self._lock:
            start = _tool_pose(self._robot["hands"][side])
            # The right hand is carried from where it is commanded to hold, which a loaded hand sags below.
            carried = self._hand_target["right"]
        if not all(
            lo <= v <= hi
            for v, lo, hi in zip(start["position_m"], limits.workspace_min_m, limits.workspace_max_m, strict=True)
        ):
            detail = "the hand starts outside the workspace, so a straight segment from it cannot be commanded"
            self._report(
                activity_id, ActivityStatus("failed", {"start": start}, error="start_outside_workspace", detail=detail)
            )
            return
        elapsed, follower_error = 0.0, 0.0
        try:
            while True:
                if self._skip(activity_id):
                    return
                dt = POSE_STEP_S if elapsed < judge.duration_s else SETTLE_STEP_S
                elapsed += dt
                position, xyzw = judge.reference(elapsed)
                targets = {side: {"frame_id": "world", "position_m": position, "orientation_xyzw": xyzw}}
                if both:
                    targets["right"] = _carry(carried, judge.start_m, judge.start_xyzw, position, xyzw)
                with self._lock:
                    self._hand_target.update(targets)
                hands = {
                    s: {
                        "position": t["position_m"],
                        "quaternion": [t["orientation_xyzw"][3], *t["orientation_xyzw"][:3]],
                    }
                    for s, t in targets.items()
                }
                reply = self._act({"action": "step", "seconds": dt, "hands": hands})
                if reply["robot"]["fallen"]:
                    self._report(activity_id, ActivityStatus("failed", error="robot_fell", detail="the robot fell"))
                    return
                measured = _tool_pose(reply["robot"]["hands"][side])
                status = judge.update(elapsed, measured["position_m"], measured["orientation_xyzw"], dt)
                if both and status.phase != "failed":
                    off = math.dist(reply["robot"]["hands"]["right"]["position"], targets["right"]["position_m"])
                    follower_error = max(follower_error, off)
                    if off > limits.path_tolerance_m:
                        detail = "the right hand fell behind where it was carried; the held object may have slipped"
                        status = ActivityStatus(
                            "failed",
                            {**status.measurements, "right_error_m": off, "at_s": elapsed},
                            error="path_tolerance_violated",
                            detail=detail,
                        )
                if status.phase == "failed":
                    self._report(activity_id, status)
                    self._fallback(rids)  # the hands hold where they were measured, not the unreached targets
                    return
                if status.phase == "done":
                    break
                if elapsed >= judge.duration_s:
                    self._report(activity_id, ActivityStatus("finishing"))
        except _Refused as refusal:
            self._report(activity_id, ActivityStatus("failed", error="task_rejected", detail=str(refusal)))
            return
        with self._lock:
            hand = self._robot["hands"][side]
            for rid in rids:
                self._settled[rid] = True  # the judge measured it within tolerance for settle_time_s
        status.measurements.update(
            touch_normal_force_N=hand["touch_normal_force_N"], controller_status=hand["controller_status"]
        )
        if both:
            status.measurements["peak_right_error_m"] = follower_error
        self._report(activity_id, status)

    def _posture(self, activity_id: str, judge: PostureJudge) -> None:
        """One `grip` action to the posture's opening, then steps at rest. The judge decides from the finger
        angles measured after each step."""
        if self._skip(activity_id):
            return
        action, limits = judge.action, judge.limits
        try:
            self._act({"action": "grip", "hand": "right", "opening": POSTURE_OPENING[action.posture]})
            self._report(activity_id, ActivityStatus("finishing"))
            rested, status = self._rest(
                ["right_hand"],
                activity_id=activity_id,
                until=lambda robot: judge.update(
                    {f"{name}": angle for name, angle in robot["hands"]["right"]["finger_joint_angles"].items()},
                    SETTLE_STEP_S,
                ),
            )
        except _Refused as refusal:
            self._report(activity_id, ActivityStatus("failed", error="task_rejected", detail=str(refusal)))
            return
        with self._lock:
            hand = self._robot["hands"]["right"]
            self._settled["right_hand"] = rested >= SETTLE_S
        if status.phase != "done":
            targets = limits.postures[action.posture]
            error = max(abs(hand["finger_joint_angles"][name] - angle) for name, angle in targets.items())
            detail = "the fingers stopped short of the posture; something may be in the hand"
            status = ActivityStatus("failed", {"max_error_rad": error}, error="posture_not_reached", detail=detail)
        status.measurements.update(touch_normal_force_N=hand["touch_normal_force_N"], at_rest=rested >= SETTLE_S)
        self._report(activity_id, status)

    def _fallback(
        self, resources: Sequence[str], carry: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    ) -> None:
        """What stop() queues: no travel, the gripper held where it is measured to be, and each arm held at its
        hand's measured pose; or, given the carrier of a stopped base goal that carried the hand targets, both arms
        held at their carried targets, still carried with the base while it comes to rest."""
        with self._lock:
            if self._finished:
                return
        try:
            rested, _ = self._rest(resources, hold_measured=True, carry=carry)
            settled = rested >= SETTLE_S
        except _Refused as refusal:  # a fallen robot or a finished episode accepts no further motion
            log.warning("the task simulator refused the fallback: %s", refusal)
            settled = False
        with self._lock:
            for rid in resources:
                self._settled[rid] = settled

    def _rest(
        self,
        resources: Sequence[str],
        *,
        hold_measured: bool = False,
        activity_id: str | None = None,
        until: Callable[[dict[str, Any]], ActivityStatus] | None = None,
        carry: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> tuple[float, ActivityStatus]:
        """Step with zero travel until measurements show these resources at rest and `until`, if given, decides
        "done" from the robot state after a step; give up after SETTLE_LIMIT_S. Returns how long the resources
        were last measured at rest, and `until`'s last decision.

        hold_measured: first set the left gripper and the arms among `resources` to hold where they are measured
        (arms that `carry` commands hold their carried targets instead).
        The jaws are never set wider than they were asked to be: jaws stopped on an object keep squeezing it,
        since a position-controlled gripper held at its measured gap presses with no force and drops what it holds.
        The base is at rest when the task reports a stable stand; a gripper, hand or arm when it stops moving.
        carry: hand targets for each step from the robot state before it, for arms carried with the base.
        """
        with self._lock:
            before = self._robot
        request: dict[str, Any] = {"action": "step", "seconds": SETTLE_STEP_S}
        if hold_measured:
            if "left_gripper" in resources:
                left = before["hands"]["left"]
                request["openings"] = {"left": min(1.0, left["requested_opening"], left["gap_m"] / MAX_GAP_M)}
            hands = {}
            for side in SIDES:
                if f"{side}_arm" in resources and carry is None:
                    pose = _tool_pose(before["hands"][side])
                    with self._lock:
                        self._hand_target[side] = pose
                    hands[side] = {
                        "position": pose["position_m"],
                        "quaternion": [pose["orientation_xyzw"][3], *pose["orientation_xyzw"][:3]],
                    }
            if hands:
                request["hands"] = hands
        at_rest_for, waited, status = 0.0, 0.0, ActivityStatus("finishing")
        while (at_rest_for < SETTLE_S or (until is not None and status.phase != "done")) and waited < SETTLE_LIMIT_S:
            if activity_id is not None and self._skip(activity_id):
                raise _Cancelled
            if carry is not None:
                request["hands"] = carry(before)
            robot = self._act(request)["robot"]
            waited += SETTLE_STEP_S
            resting = _at_rest(resources, before, robot, self._arm_still_m)
            before = robot
            at_rest_for = at_rest_for + SETTLE_STEP_S if resting else 0.0
            if until is not None:
                status = until(robot)
        return at_rest_for, status


def _at_rest(resources: Sequence[str], before: dict[str, Any], after: dict[str, Any], arm_still_m: float) -> bool:
    """Whether each resource was still over one settle step, measured from the robot state before and after."""
    hb, ha = before["hands"], after["hands"]
    checks = {
        "base": lambda: after["support"]["stable"],
        "left_gripper": lambda: abs(ha["left"]["gap_m"] - hb["left"]["gap_m"]) < GAP_STILL_M,
        "left_arm": lambda: math.dist(ha["left"]["position"], hb["left"]["position"]) < arm_still_m,
        "right_arm": lambda: math.dist(ha["right"]["position"], hb["right"]["position"]) < arm_still_m,
        "right_hand": lambda: (
            max(
                abs(angle - hb["right"]["finger_joint_angles"][name])
                for name, angle in ha["right"]["finger_joint_angles"].items()
            )
            < FINGER_STILL_RAD
        ),
    }
    return all(checks[rid]() for rid in resources)


def _carry(
    pose: dict[str, Any],
    start_m: Sequence[float],
    start_xyzw: Sequence[float],
    now_m: Sequence[float],
    now_xyzw: Sequence[float],
) -> dict[str, Any]:
    """`pose` moved by the rigid motion that takes (start_m, start_xyzw) to (now_m, now_xyzw)."""
    turn = _qmul(now_xyzw, _qconj(start_xyzw))
    offset = _rotate(turn, [p - s for p, s in zip(pose["position_m"], start_m, strict=True)])
    return {
        "frame_id": "world",
        "position_m": [n + o for n, o in zip(now_m, offset, strict=True)],
        "orientation_xyzw": _qmul(turn, pose["orientation_xyzw"]),
    }


def _qmul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """The product a * b of two xyzw quaternions: b's rotation, then a's."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _qconj(q: Sequence[float]) -> list[float]:
    return [-q[0], -q[1], -q[2], q[3]]


def _rotate(q: Sequence[float], v: Sequence[float]) -> list[float]:
    return _qmul(_qmul(q, [*v, 0.0]), _qconj(q))[:3]


def _flat_pose(pose: dict[str, Any]) -> dict[str, float]:
    """A pose as the flat numbers a retained target records."""
    (x, y, z), (qx, qy, qz, qw) = pose["position_m"], pose["orientation_xyzw"]
    return {"x_m": x, "y_m": y, "z_m": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}


def _tool_pose(hand: dict[str, Any]) -> dict[str, Any]:
    """A hand's tool site as the task measures it, with the quaternion reordered from the task's w, x, y, z."""
    w, x, y, z = hand["quaternion_wxyz"]
    return {"frame_id": "world", "position_m": hand["position"], "orientation_xyzw": [x, y, z, w]}


def _camera_image(metadata: dict[str, Any], clock_epoch: str) -> CameraImage:
    """The rgb image a task capture describes, in CameraImage's conventions.

    The task states its lens as one focal length in pixels, centred, and its camera's rotation in MuJoCo's camera
    axes (y up, looking along -z); optical axes flip y and z.
    """
    width, height, focal = metadata["width"], metadata["height"], float(metadata["focal_px"])
    optical = [[row[0], -row[1], -row[2]] for row in metadata["world_from_camera"]]
    return CameraImage(
        kind="rgb",
        media_type="image/png",
        width=width,
        height=height,
        clock_id="task_sim",
        clock_epoch=clock_epoch,
        capture_time_ns=str(round(metadata["sim_time"] * 1e9)),
        focal_px=[focal, focal],
        principal_px=[width / 2, height / 2],
        camera_pose=Pose(
            frame_id="world",
            position_m=[float(v) for v in metadata["camera_position"]],
            orientation_xyzw=_quaternion_xyzw(optical),
        ),
    )


def _quaternion_xyzw(m: list[list[float]]) -> list[float]:
    """The unit quaternion of a rotation matrix, x, y, z, w; from its largest component, for precision."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 2 * math.sqrt(1 + trace)
        x, y, z, w = (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, s / 4
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2 * math.sqrt(1 + m[0][0] - m[1][1] - m[2][2])
        x, y, z, w = s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s
    elif m[1][1] > m[2][2]:
        s = 2 * math.sqrt(1 + m[1][1] - m[0][0] - m[2][2])
        x, y, z, w = (m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s
    else:
        s = 2 * math.sqrt(1 + m[2][2] - m[0][0] - m[1][1])
        x, y, z, w = (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4, (m[1][0] - m[0][1]) / s
    return [x, y, z, w]
