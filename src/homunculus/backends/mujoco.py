"""A MuJoCo backend: an arm and an optional parallel gripper, driven through their actuators.

Physics runs in real time on the runtime's clock. On each physics step every
control group's controller writes position references to its actuators, and the
profile's judge (profiles.py) decides success from the measured joint state, so
a result here is evidence about simulated physics. It is not evidence about any
physical robot.

It serves a rig (rigs/__init__.py): a folder whose rig.toml names an MJCF model,
the joints to drive, and the robot's cameras. By default that is `panda`, the
Franka Emika Panda from MuJoCo Menagerie at a table with a scanned mug and bowl.
A rig's model works when:

- each arm joint is a limited hinge or slide joint, and each gripper joint a
  limited slide joint;
- each of those joints is moved by exactly one position actuator: fixed gain
  and affine bias, as `<position>` builds and Menagerie's `<general>` actuators
  are, acting on the joint or on a fixed tendon over the joints (the Panda's two
  fingers share one actuator on a tendon this way);
- the model has a keyframe named `home`, where the world starts and restarts.

The gripper's width is the sum of its finger joints' positions.

The robot's cameras are the model cameras that the rig's `cameras` names;
observations record them. Every other named camera is a spectator camera, for
people watching its live video. The Panda rig's own cameras are `wrist`, on the
hand, and `workspace`, across the table; its spectator camera is `overview`.
Cameras are set by their vertical field of view. Frames are rendered by MuJoCo's
OpenGL renderer and sent as JPEG, with the lens and the camera's pose at capture.

check() is how a rig is accepted: it drives one through the runtime, as a client
would, and reports what was measured.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import secrets
import tempfile
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import mujoco
import mujoco_menagerie
import numpy as np
from PIL import Image

from .. import rigs
from ..journal import Journal
from ..models import (
    TERMINAL_GOAL_STATES,
    BackendInfo,
    CameraImage,
    ClockInfo,
    Goal,
    GoalSubmit,
    GripperWidthGoal,
    GripperWidthLimits,
    JointLimit,
    JointTrajectoryGoal,
    JointTrajectoryLimits,
    Pose,
    ProfileBinding,
    RobotError,
    Sample,
    SensorInfo,
    SessionOpen,
    TrajectoryPoint,
)
from ..profiles import ActivityStatus, TrajectoryJudge, WidthJudge, sample_trajectory
from ..runtime import (
    Backend,
    BackendFacts,
    Clock,
    Feedback,
    Frame,
    GoalCommand,
    Principal,
    ResourceSpec,
    Runtime,
    StartupError,
    SystemClock,
)

PERIOD_S = 0.01
# Physics never runs more than this far behind the host clock; beyond it, time is dropped and counted as lag.
MAX_CATCH_UP_S = 0.1
VIDEO_FPS = 20.0
JPEG_QUALITY = 90
# The files of a Menagerie robot that are about it, not part of its model. They stay out of the folder a rig is
# compiled in, so a rig may have a README and a LICENSE of its own.
MENAGERIE_DOCUMENTS = ("README.md", "LICENSE", "CHANGELOG.md")
# MuJoCo's cameras look along their -z with y up; optical axes look along +z with y down.
OPTICAL = np.diag([1.0, -1.0, -1.0])


def robots(rig: str = "panda", robot_id: str | None = None) -> dict[str, Backend]:
    """The robot `--backend mujoco` serves.

    Options: `rig`, the name of a rig that ships with homunculus or the path of a rig folder (default: the Panda
    at a table); `robot_id` (default: the rig folder's name).
    """
    folder = rigs.find(rig)
    return {robot_id or folder.name: MujocoBackend(SystemClock(), folder)}


def _compile(folder: Path, rig: rigs.Rig) -> tuple[mujoco.MjModel, dict[str, str]]:
    """The rig's model, compiled, and the revisions that say what it was compiled from.

    A model can only `<include>` a robot that sits in its own folder: MuJoCo looks for an included file's meshes
    from the including file. So the rig's files and its Menagerie robot's files are set side by side, as links,
    in a folder that lasts for the compile. Nothing is written into the rig or into Menagerie's download.

    A name that starts with a dot is not part of a rig: `.git`, or the `.DS_Store` that macOS leaves in a folder.
    """
    with tempfile.TemporaryDirectory(prefix="homunculus-rig-") as temporary:
        side_by_side = Path(temporary)
        for entry in folder.iterdir():
            if not entry.name.startswith("."):
                (side_by_side / entry.name).symlink_to(entry)
        source = {}
        if rig.menagerie is not None:
            try:
                robot = mujoco_menagerie.get(rig.menagerie)
                files = robot.path()  # downloaded and checked on first use, then read from Menagerie's cache
            except mujoco_menagerie.MenagerieError as error:
                raise StartupError(
                    f"cannot get the robot {rig.menagerie!r} from MuJoCo Menagerie: {error}\n"
                    "`homunculus install` downloads a rig's robot while this machine is online"
                ) from None
            source = {"menagerie": f"{robot.name}@{robot.oid[:12]}"}
            for entry in files.iterdir():
                if entry.name in MENAGERIE_DOCUMENTS or entry.name.startswith("."):
                    continue
                if (side_by_side / entry.name).exists():
                    raise StartupError(
                        f"{entry.name} is in the rig folder {folder} and in Menagerie's {rig.menagerie}, "
                        "which load side by side; rename the rig's"
                    )
                (side_by_side / entry.name).symlink_to(entry)
        # `model_sha256` is over every file beside the model, by its name there and its bytes, so it is the same
        # on every machine and changes when any file the model includes does. The compiled model would not do:
        # it records the path of this temporary folder. A file the model names by an absolute path is not in it.
        digest = hashlib.sha256()
        for directory, subdirectories, names in side_by_side.walk(follow_symlinks=True):
            subdirectories[:] = sorted(name for name in subdirectories if not name.startswith("."))
            for name in sorted(name for name in names if not name.startswith(".")):
                content = (directory / name).read_bytes()
                beside = (directory / name).relative_to(side_by_side).as_posix()
                digest.update(f"{beside}\0{len(content)}\0".encode() + content)
        revisions = {"model_sha256": digest.hexdigest()[:12], **source}
        try:
            spec = mujoco.MjSpec.from_file(str(side_by_side / rig.model))
            for mount in rig.mount:
                # MJCF cannot add a child to a body that an included file defines, so a rig states such a camera.
                body = spec.body(mount.body)
                if body is None:
                    raise StartupError(
                        f"rig.toml mounts the camera {mount.camera!r} on the body {mount.body!r}, "
                        f"which the model {rig.model} does not have"
                    )
                body.add_camera(name=mount.camera, pos=mount.pos, quat=mount.quat, fovy=mount.fovy)
            return spec.compile(), revisions
        except ValueError as error:
            raise StartupError(f"cannot load the model {rig.model} of the rig {folder}: {error}") from None


@dataclass
class _Motion:
    activity_id: str
    started: float
    columns: list[int]  # which of the group's joints the action commands
    judge: TrajectoryJudge | WidthJudge  # holds the goal's action and limits


@dataclass
class _Group:
    """One control group: where its joints live in the model, and its controller's state."""

    spec: ResourceSpec
    qpos: np.ndarray  # indices into data.qpos
    dof: np.ndarray  # indices into data.qvel
    actuators: np.ndarray  # indices into data.ctrl
    moves: np.ndarray  # moves[a, j]: how far actuator a's length changes per unit of joint j
    gain: np.ndarray  # each actuator's gain on ctrl
    bias: np.ndarray  # each actuator's force at zero ctrl: bias0 + bias1*length + bias2*velocity
    low: np.ndarray  # lower joint limits
    still_speed: float
    settle_time_s: float
    reference: np.ndarray  # the position each joint is asked to hold or follow
    motion: _Motion | None = None
    still_for: float = 0.0


class MujocoBackend(Backend):
    reports_goal_samples = True

    def __init__(self, clock: Clock, folder: Path):
        """`folder` is a rig: rig.toml and the model it names."""
        rig = rigs.load(folder)
        arm, gripper, cameras = rig.arm, rig.gripper, rig.cameras
        self._model, revisions = _compile(folder, rig)
        name = self._model.names.split(b"\0", 1)[0].decode()  # <mujoco model="...">
        self._revisions = {"robot_description": name, **revisions}
        self._notice = (
            f"MuJoCo simulation of the model {name!r}, from the rig {folder.name!r}. "
            "Results are simulated physics, not a physical robot."
        )
        self._data = mujoco.MjData(self._model)
        self._clock = clock
        self._last: float | None = None  # host time of the previous advance; the world starts at the first
        self._behind = 0.0  # host time not yet simulated
        self._lag = 0.0  # host time this world never simulated, in total
        # A world lives in this process and names its clock's epoch, so the name must not repeat after a restart.
        self._instance = secrets.token_hex(4)
        self._world = 0
        try:
            self._home = self._model.key("home").id
        except KeyError:
            raise StartupError("the MuJoCo model needs a keyframe named 'home' to start and reset from") from None

        model = self._model
        named = [model.camera(i).name for i in range(model.ncam) if model.camera(i).name]
        for camera in cameras:
            if camera not in named:
                raise StartupError(
                    f"the rig's model has no camera named {camera!r}; it has {named}. rig.toml's `cameras` names "
                    "the robot's own, and each `[[mount]]` adds one"
                )
        for camera in named:
            if model.cam_sensorsize[model.camera(camera).id].any():
                raise StartupError(f"camera {camera!r} sets a sensor size; only cameras set by fovy are supported")
        self._cameras = {name: "camera" if name in cameras else "spectator_camera" for name in named}
        # Captures render on one thread of their own. The renderer's OpenGL context stays locked to the thread
        # that first used it: used, or even freed, from another thread, it waits for that one forever.
        self._render_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-camera")
        self._renderer: mujoco.Renderer | None = None  # made by the first capture, on that thread
        self._shown = mujoco.MjData(model)  # the copy of the world a capture renders
        # Held by the control thread while it changes the world, and by a capture while it copies the world.
        self._stepping = threading.Lock()

        hinge, slide = mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
        arm_joints = [self._joint(name, (hinge, slide)) for name in arm]
        self._arm_limits = JointTrajectoryLimits(
            profile="joints.trajectory/1",
            joints=[
                JointLimit(
                    name=name,
                    unit="rad" if model.jnt_type[jid] == hinge else "m",
                    min_position=float(model.jnt_range[jid, 0]),
                    max_position=float(model.jnt_range[jid, 1]),
                    max_velocity=1.5,
                    max_acceleration=6.0,
                )
                for name, jid in zip(arm, arm_joints, strict=True)
            ],
            max_points=64,
            max_duration_s=30.0,
            start_position_tolerance=0.05,
            start_velocity_tolerance=0.1,
            path_tolerance=0.1,
            goal_tolerance=0.02,
            settle_velocity=0.05,
            settle_time_s=0.2,
        )
        self._groups = {
            "arm": self._group("arm", arm_joints, self._arm_limits.settle_velocity, self._arm_limits.settle_time_s)
        }
        self._gripper_limits: GripperWidthLimits | None = None
        if gripper:
            fingers = [self._joint(name, (slide,)) for name in gripper]
            ranges = model.jnt_range[fingers]
            self._gripper_limits = GripperWidthLimits(
                profile="gripper.width/1",
                min_width_m=float(ranges[:, 0].sum()),
                max_width_m=float(ranges[:, 1].sum()),
                speed_m_s=0.05,
                goal_tolerance_m=0.003,
                settle_time_s=0.2,
                # No contact or force sensing is read from the fingers, so only width completion is offered.
                complete_on=["width_reached"],
            )
            self._groups["gripper"] = self._group("gripper", fingers, 0.005, self._gripper_limits.settle_time_s)
        self._reset()

    def _joint(self, name: str, kinds: Sequence[mujoco.mjtJoint]) -> int:
        """A joint this backend may drive: of an allowed kind, and limited."""
        model = self._model
        try:
            jid = model.joint(name).id
        except KeyError:
            raise StartupError(f"the MuJoCo model has no joint named {name!r}") from None
        if mujoco.mjtJoint(int(model.jnt_type[jid])) not in kinds or not model.jnt_limited[jid]:
            allowed = " or ".join(kind.name.removeprefix("mjJNT_").lower() for kind in kinds)
            raise StartupError(f"joint {name!r} must be a {allowed} joint with a range")
        return jid

    def _group(self, rid: str, joints: list[int], still_speed: float, settle_time_s: float) -> _Group:
        """A control group of joints, and the position actuators that move them."""
        model = self._model
        moves = np.zeros((model.nu, len(joints)))
        for a in range(model.nu):
            kind, target, gear = model.actuator_trntype[a], model.actuator_trnid[a, 0], model.actuator_gear[a, 0]
            if kind == mujoco.mjtTrn.mjTRN_JOINT:
                wraps = [(target, 1.0)]
            elif kind == mujoco.mjtTrn.mjTRN_TENDON:  # a fixed tendon's length is a weighted sum of its joints
                first = model.tendon_adr[target]
                wraps = [
                    (model.wrap_objid[w], model.wrap_prm[w])
                    for w in range(first, first + model.tendon_num[target])
                    if model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT
                ]
            else:
                continue
            for jid, coefficient in wraps:
                if jid in joints:
                    moves[a, joints.index(jid)] = gear * coefficient
        for column, jid in enumerate(joints):
            movers = np.flatnonzero(moves[:, column])
            if len(movers) != 1 or not _is_position_servo(model, movers[0]):
                raise StartupError(
                    f"joint {model.joint(jid).name!r} must be moved by exactly one position actuator, "
                    "on the joint or on a fixed tendon"
                )
        actuators = np.flatnonzero(moves.any(axis=1))
        return _Group(
            spec=ResourceSpec(
                id=rid, role=rid, joints=[model.joint(j).name for j in joints], controller=f"mujoco_{rid}_position"
            ),
            qpos=model.jnt_qposadr[joints].copy(),
            dof=model.jnt_dofadr[joints].copy(),
            actuators=actuators,
            moves=moves[actuators],
            gain=model.actuator_gainprm[actuators, 0].copy(),
            bias=model.actuator_biasprm[actuators, :3].copy(),
            low=model.jnt_range[joints, 0].copy(),
            still_speed=still_speed,
            settle_time_s=settle_time_s,
            reference=np.zeros(len(joints)),
        )

    def _reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._home)
        mujoco.mj_forward(self._model, self._data)
        for group in self._groups.values():
            group.reference = self._data.qpos[group.qpos].copy()
            group.motion, group.still_for = None, 0.0
        self._behind = self._lag = 0.0

    def _world_id(self) -> str:
        return f"mujoco-world-{self._instance}-{self._world}"

    def describe(self) -> BackendFacts:
        bindings = [
            ProfileBinding(
                profile="joints.trajectory/1",
                target="arm",
                role="arm",
                resources=["arm"],
                modes=["goal"],
                controller="mujoco_arm_position",
                assistance=(
                    "joint position control with velocity feedforward through force-limited actuators; "
                    "no gravity compensation, no collision avoidance"
                ),
                stop_deadline_wall_s=1.0,
                limits=self._arm_limits,
            )
        ]
        if self._gripper_limits is not None:
            bindings.append(
                ProfileBinding(
                    profile="gripper.width/1",
                    target="gripper",
                    role="gripper",
                    resources=["gripper"],
                    modes=["goal"],
                    controller="mujoco_gripper_position",
                    assistance="the fingers are driven together toward the requested gap at a fixed speed",
                    stop_deadline_wall_s=1.0,
                    limits=self._gripper_limits,
                )
            )
        return BackendFacts(
            info=BackendInfo(
                kind="mujoco",
                physics="simulated",
                notice=self._notice,
                versions={"mujoco": mujoco.__version__},
            ),
            world_id=self._world_id(),
            mode="position_control",
            clock_mode="realtime",
            clocks=[
                ClockInfo(
                    id="mujoco_sim",
                    epoch=self._world_id(),
                    description=(
                        f"simulation time in {self._model.opt.timestep:g} s steps, advanced in real time; "
                        "zero at world reset"
                    ),
                )
            ],
            frames=["world"],
            sensors=[
                *(
                    SensorInfo(
                        id=rid,
                        kind="joint_state",
                        provenance="simulation_ground_truth",
                        description=(
                            f"joint positions, velocities, and the force or torque the actuators apply at each "
                            f"joint of {rid}, read from the simulator"
                        ),
                    )
                    for rid in self._groups
                ),
                *(
                    SensorInfo(
                        id=name,
                        kind=kind,
                        provenance="simulation_ground_truth",
                        description=(
                            f"{self._model.vis.global_.offwidth}x{self._model.vis.global_.offheight} RGB as JPEG, "
                            f"rendered from the model's camera {name!r}"
                        ),
                        video_fps=VIDEO_FPS,
                    )
                    for name, kind in self._cameras.items()
                ),
            ],
            revisions=self._revisions,
            resources=[group.spec for group in self._groups.values()],
            bindings=bindings,
            concurrent=[list(self._groups)] if len(self._groups) > 1 else [],
            management=["recover", "world.reset"],
            period_s=PERIOD_S,
        )

    def apply(self, activity_id: str, contract: GoalCommand) -> None:
        group = self._groups[contract.resources[0]]
        action = contract.action
        if isinstance(action, JointTrajectoryGoal):
            columns = [group.spec.joints.index(name) for name in action.joints]
        else:
            columns = list(range(len(group.spec.joints)))
        group.motion = _Motion(activity_id, self._data.time, columns, contract.judge)

    def stop(self, resources: Sequence[str]) -> None:
        for rid in resources:
            self._hold_measured(self._groups[rid])

    def _hold_measured(self, group: _Group) -> None:
        """The fallback: drop the motion and hold where the joints are measured to be."""
        group.motion = None
        group.reference = self._data.qpos[group.qpos].copy()
        group.still_for = 0.0  # standstill has to be measured again after the command

    def manage(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
        if operation == "recover":
            for rid in parameters["resources"]:
                self._hold_measured(self._groups[rid])
            return {}
        if operation == "world.reset":
            with self._stepping:  # a capture shows the old world or the new one, each under its own world ID
                self._world += 1
                self._reset()
            return {"world_id": self._world_id()}
        raise RobotError("unsupported_operation", f"the MuJoCo backend has no {operation}", operation=operation)

    def capture(self, sensor: str) -> list[Frame]:
        return [self._render_thread.submit(self._render, sensor).result()]

    def _render(self, camera: str) -> Frame:
        """On the render thread: the world as the camera sees it now."""
        model = self._model
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(model, model.vis.global_.offheight, model.vis.global_.offwidth)
            except Exception as error:  # no OpenGL context could be made in this process
                raise RobotError(
                    "backend_error",
                    f"MuJoCo cannot render camera {camera!r} here: {error} (the MUJOCO_GL environment variable "
                    "chooses its OpenGL, such as egl on a machine without a display)",
                ) from None
        with self._stepping:
            mujoco.mj_copyData(self._shown, model, self._data)
            world = self._world_id()
        shown, cid = self._shown, model.camera(camera).id
        self._renderer.update_scene(shown, camera=camera)
        pixels = self._renderer.render()
        height, width, _ = pixels.shape
        encoded = BytesIO()
        Image.fromarray(pixels).save(encoded, format="JPEG", quality=JPEG_QUALITY)
        focal = height / 2 / math.tan(math.radians(model.cam_fovy[cid]) / 2)  # square pixels
        wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(wxyz, (shown.cam_xmat[cid].reshape(3, 3) @ OPTICAL).flatten())
        image = CameraImage(
            kind="rgb",
            media_type="image/jpeg",
            width=width,
            height=height,
            clock_id="mujoco_sim",
            clock_epoch=world,
            capture_time_ns=str(round(shown.time * 1e9)),
            focal_px=[focal, focal],
            principal_px=[width / 2, height / 2],
            camera_pose=Pose(
                frame_id="world",
                position_m=shown.cam_xpos[cid].tolist(),
                orientation_xyzw=[*wxyz[1:].tolist(), float(wxyz[0])],
            ),
        )
        return Frame(image=image, data=encoded.getvalue())

    def close(self) -> None:
        def free() -> None:
            if self._renderer is not None:
                self._renderer.close()

        # Freed on the render thread, after the captures already asked for: from any other thread (the garbage
        # collector's at exit, for one) the renderer would wait forever for its OpenGL context.
        self._render_thread.submit(free)
        self._render_thread.shutdown()

    def advance(self) -> Feedback:
        """Step physics up to the host clock, running every group's controller on each step."""
        model, data = self._model, self._data
        now = self._clock.now()
        if self._last is not None:
            self._behind += max(0.0, now - self._last)
        self._last = now
        if self._behind > MAX_CATCH_UP_S:
            self._lag += self._behind - MAX_CATCH_UP_S
            self._behind = MAX_CATCH_UP_S
        step = model.opt.timestep
        with self._stepping:  # a capture copies the world between steps, never halfway through one
            for _ in range(int(self._behind / step)):
                for group in self._groups.values():
                    self._control(group, step)
                mujoco.mj_step(model, data)
                self._behind -= step

        feedback = Feedback(execution_time_s=data.time, samples={}, lag_s=self._lag)
        for rid, group in self._groups.items():
            names = group.spec.joints
            positions, velocities = data.qpos[group.qpos], data.qvel[group.dof]
            values: dict[str, Any] = {
                "positions": dict(zip(names, positions.tolist(), strict=True)),
                "velocities": dict(zip(names, velocities.tolist(), strict=True)),
                "actuator_forces": dict(zip(names, data.qfrc_actuator[group.dof].tolist(), strict=True)),
            }
            if rid == "gripper":
                values["width_m"] = float(positions.sum())
            feedback.samples[rid] = Sample(
                source=f"mujoco:{rid}",
                provenance="simulation_ground_truth",
                clock_id="mujoco_sim",
                clock_epoch=self._world_id(),
                capture_time_ns=str(round(data.time * 1e9)),
                valid=True,
                values=values,
            )
            if group.motion is not None:
                status = group.motion.judge.status or ActivityStatus("executing")
                feedback.activities[group.motion.activity_id] = status
                if status.phase == "done":
                    group.motion = None  # the reference it reached stays in place as the hold
            feedback.stopped[rid] = group.motion is None and group.still_for >= group.settle_time_s
            feedback.holding[rid] = dict(zip(names, group.reference.tolist(), strict=True))
        return feedback

    def _control(self, group: _Group, step: float) -> None:
        """One physics step of one group: update the reference, command the actuators, judge the motion."""
        data = self._data
        positions, velocities = data.qpos[group.qpos], data.qvel[group.dof]
        rates = np.zeros(len(group.reference))
        motion = group.motion
        status = motion.judge.status if motion is not None else None
        if motion is not None and (status is None or status.phase in ("executing", "finishing")):
            judge = motion.judge
            if isinstance(judge, TrajectoryJudge):
                # joints.trajectory/1: follow the validated path.
                elapsed = data.time - motion.started
                reference, wanted = sample_trajectory(judge.action.points, elapsed)
                columns = motion.columns
                group.reference[columns] = reference
                status = judge.update(elapsed, positions[columns].tolist(), velocities[columns].tolist(), step)
                if status.phase != "failed":
                    rates[columns] = wanted
            else:
                # gripper.width/1: walk every finger's reference toward an equal share of the gap.
                fingers = len(group.reference)
                share = group.low + (judge.action.width_m - group.low.sum()) / fingers
                reach = judge.limits.speed_m_s / fingers * step
                group.reference += np.clip(share - group.reference, -reach, reach)
                judge.update(float(positions.sum()), step)
        # Each actuator's force is gain*ctrl + bias0 + bias1*length + bias2*velocity. This ctrl turns it into a
        # spring and damper on the error from the reference length and velocity, so the damping acts on the
        # velocity error instead of on the motion itself.
        lengths, speeds = group.moves @ group.reference, group.moves @ rates
        bias0, bias1, bias2 = group.bias.T
        data.ctrl[group.actuators] = -(bias0 + bias1 * lengths + bias2 * speeds) / group.gain
        still = bool(np.all(np.abs(velocities) <= group.still_speed))
        group.still_for = group.still_for + step if still else 0.0


def _is_position_servo(model: mujoco.MjModel, actuator: int) -> bool:
    """gain*ctrl + bias0 + bias1*length + bias2*velocity, pulling back toward a length that ctrl sets: what
    `<position>` builds. The controller relies on that form."""
    return bool(
        model.actuator_gaintype[actuator] == mujoco.mjtGain.mjGAIN_FIXED
        and model.actuator_biastype[actuator] == mujoco.mjtBias.mjBIAS_AFFINE
        and model.actuator_gainprm[actuator, 0] > 0
        and model.actuator_biasprm[actuator, 1] < 0
    )


# --------------------------------------------------------------------------
# Checking a rig
# --------------------------------------------------------------------------

HOLD_S = 2.0  # how long the start pose must hold
MOVE_S = 1.5  # how long each checking move takes
MOVE = 0.15  # how far each checking move goes, in the joint's unit: at most a fifth of the joint's range
GOAL_WAIT_S = 90.0  # longer than any goal's own deadline


@dataclass(frozen=True)
class Step:
    """One thing check() tried, and what was measured."""

    name: str
    passed: bool
    detail: str


class _SteppedClock:
    """Time that moves only when the check moves it, so a check takes as long as its physics takes to compute."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t


def check(folder: Path, images: Path) -> list[Step]:
    """Drive a rig as a client would, through the runtime and its judges, and say what was measured.

    A rig that cannot be loaded raises StartupError, as it does when a service starts it. One that loads is
    asked to hold its start pose, to move each arm joint to a goal and back, to close and open its gripper, and
    to take a picture with each of its cameras, saved into `images`.

    Passing shows that the rig is wired correctly: its joints, actuators, start pose, and cameras are what
    rig.toml says. It does not show that the arm is strong enough for a task: an SO-101 with one actuator at a
    hundredth of its stiffness still passes. It does not show that a camera is aimed well either: a camera
    inside a mesh still renders a picture. Look at the pictures.
    """
    clock = _SteppedClock()
    steps: list[Step] = []
    with (
        MujocoBackend(clock, folder) as backend,
        tempfile.TemporaryDirectory(prefix="homunculus-check-") as temporary,
        Journal(Path(temporary) / "journal.sqlite", "boot-check") as journal,
    ):
        runtime = Runtime(folder.name, backend, journal, boot_id="boot-check", clock=clock)
        client = Principal("check")
        context = runtime.describe().context_id
        limits = {binding.profile: binding.limits for binding in runtime.describe().profiles}
        arm = limits["joints.trajectory/1"]
        requests = itertools.count()

        def run(seconds: float) -> None:
            for _ in range(round(seconds / PERIOD_S)):
                clock.t += PERIOD_S
                runtime.tick()
                journal.flush()

        def positions() -> dict[str, float]:
            return runtime.observe().samples["arm"].values["positions"]

        def finish(action: JointTrajectoryGoal | GripperWidthGoal) -> Goal:
            """Submit a goal and run until it ends by its own rules: success, a tolerance, or its deadline."""
            request = GoalSubmit(
                request_id=f"check-{next(requests)}", session_id=session, context_id=context, action=action
            )
            goal_id = runtime.submit_goal(client, request).goal_id
            waited = 0.0
            while runtime.goal(goal_id).state not in TERMINAL_GOAL_STATES and waited < GOAL_WAIT_S:
                run(0.25)
                waited += 0.25
            run(0.25)  # the resources settle before the next goal asks for them
            return runtime.goal(goal_id)

        def move(joint: str, position: float) -> Goal:
            point = TrajectoryPoint(time_s=MOVE_S, positions=[position], velocities=[0.0])
            return finish(
                JointTrajectoryGoal(profile="joints.trajectory/1", target="arm", joints=[joint], points=[point])
            )

        def outcome(goal: Goal) -> str:
            return goal.state if goal.result is None else f"{goal.state} ({goal.result.reason})"

        units = {limit.name: limit.unit for limit in arm.joints}
        home = positions()
        run(HOLD_S)
        held = positions()
        worst = max(home, key=lambda joint: abs(held[joint] - home[joint]))
        drift = abs(held[worst] - home[worst])
        steps.append(
            Step(
                "holds its start pose",
                drift <= arm.goal_tolerance,
                f"{worst} moved {drift:.4f} {units[worst]} in {HOLD_S:g} s; {arm.goal_tolerance:g} is allowed",
            )
        )
        if not steps[-1].passed:
            return steps  # nothing after this would start from where it says it starts

        session = runtime.open_session(
            client, SessionOpen(request_id=f"check-{next(requests)}", context_id=context)
        ).session_id
        failed = []
        for limit in arm.joints:
            start = positions()[limit.name]
            step = min(MOVE, 0.2 * (limit.max_position - limit.min_position))
            toward_middle = 1 if start < (limit.min_position + limit.max_position) / 2 else -1
            out = move(limit.name, start + toward_middle * step)
            back = move(limit.name, home[limit.name])
            if out.state != "succeeded" or back.state != "succeeded":
                failed.append(f"{limit.name}: out {outcome(out)}, back {outcome(back)}")
        steps.append(
            Step(
                "each arm joint reaches a goal and returns",
                not failed,
                "; ".join(failed) or f"{len(arm.joints)} joints, each moved up to {MOVE:g} rad or m and back",
            )
        )

        if "gripper.width/1" in limits:
            gripper = limits["gripper.width/1"]
            widths = [(gripper.min_width_m + gripper.max_width_m) / 2, gripper.max_width_m]
            goals = [
                finish(GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=width)) for width in widths
            ]
            steps.append(
                Step(
                    "the gripper closes halfway and opens",
                    all(goal.state == "succeeded" for goal in goals),
                    ", ".join(f"{width:.3f} m {outcome(goal)}" for width, goal in zip(widths, goals, strict=True)),
                )
            )

        for sensor in runtime.describe().sensors:
            if sensor.kind != "camera":
                continue
            try:
                image = runtime.observe([sensor.id]).images[0]
            except RobotError as error:
                steps.append(Step(f"camera {sensor.id} takes a picture", False, error.message))
                continue
            images.mkdir(parents=True, exist_ok=True)
            path = images / f"{folder.name}-{sensor.id}.jpg"
            path.write_bytes(runtime.image(image.handle)[1])
            steps.append(
                Step(f"camera {sensor.id} takes a picture", True, f"{image.width}x{image.height}, saved as {path}")
            )
        return steps
