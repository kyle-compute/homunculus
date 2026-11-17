"""A MuJoCo backend: an arm and an optional parallel gripper, driven through their actuators.

Physics runs in real time on the runtime's clock. On each physics step every
control group's controller writes position references to its actuators, and the
profile's judge (profiles.py) decides success from the measured joint state, so
a result here is evidence about simulated physics. It is not evidence about any
physical robot.

By default it loads the Franka Emika Panda from MuJoCo Menagerie, copied
unchanged into assets/franka_emika_panda, standing at a table with a scanned mug
and bowl (homunculus_scene.xml there). Another model works (`--backend-arg
model=FILE`) when:

- each arm joint is a limited hinge or slide joint, and each gripper joint a
  limited slide joint;
- each of those joints is moved by exactly one position actuator: fixed gain
  and affine bias, as `<position>` builds and Menagerie's `<general>` actuators
  are, acting on the joint or on a fixed tendon over the joints (the Panda's two
  fingers share one actuator on a tendon this way);
- the model has a keyframe named `home`, where the world starts and restarts.

The gripper's width is the sum of its finger joints' positions.

The robot's cameras are the model cameras that `cameras` names; observations
record them. Every other named camera is a spectator camera, for people watching
its live video. The Panda scene's robot cameras are `wrist`, on the hand, and
`workspace`, across the table; its spectator camera is `overview`. Cameras are
set by their vertical field of view. Frames are rendered by MuJoCo's OpenGL
renderer and sent as JPEG, with the lens and the camera's pose at capture.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from ..models import (
    BackendInfo,
    CameraImage,
    ClockInfo,
    Contract,
    GripperWidthLimits,
    JointLimit,
    JointTrajectoryGoal,
    JointTrajectoryLimits,
    Pose,
    ProfileBinding,
    RobotError,
    Sample,
    SensorInfo,
)
from ..profiles import ActivityStatus, TrajectoryJudge, WidthJudge, sample_trajectory
from ..runtime import Backend, BackendFacts, Clock, Feedback, Frame, ResourceSpec, StartupError, SystemClock

PERIOD_S = 0.01
# Physics never runs more than this far behind the host clock; beyond it, time is dropped and counted as lag.
MAX_CATCH_UP_S = 0.1
VIDEO_FPS = 20.0
JPEG_QUALITY = 90
PANDA = (Path(__file__).parent.parent / "assets" / "franka_emika_panda" / "homunculus_scene.xml").resolve()
ARM_JOINTS = tuple(f"joint{n}" for n in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
PANDA_CAMERAS = ("wrist", "workspace")
# MuJoCo's cameras look along their -z with y up; optical axes look along +z with y down.
OPTICAL = np.diag([1.0, -1.0, -1.0])


def robots(
    model: str = str(PANDA),
    arm: str = ",".join(ARM_JOINTS),
    gripper: str = ",".join(FINGER_JOINTS),
    cameras: str = ",".join(PANDA_CAMERAS),
    robot_id: str = "sim-arm",
) -> dict[str, Backend]:
    """The robot `--backend mujoco` serves.

    Options: `model`, an MJCF file (default: the Panda at a table); `arm` and `gripper`, comma-separated joint
    names (an empty gripper means none); `cameras`, the model cameras that are the robot's own (the others are
    for spectators); `robot_id`.
    """
    arm_joints = [name for name in arm.split(",") if name]
    finger_joints = [name for name in gripper.split(",") if name]
    robot_cameras = [name for name in cameras.split(",") if name]
    path = Path(model).expanduser().resolve()
    return {robot_id: MujocoBackend(SystemClock(), path, arm_joints, finger_joints, robot_cameras)}


@dataclass
class _Motion:
    activity_id: str
    started: float
    columns: list[int]  # which of the group's joints the action commands
    judge: TrajectoryJudge | WidthJudge  # holds the goal's action and limits
    status: ActivityStatus = field(default_factory=lambda: ActivityStatus("executing"))


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
    def __init__(
        self, clock: Clock, model_path: Path, arm: Sequence[str], gripper: Sequence[str], cameras: Sequence[str]
    ):
        """An empty gripper means the model has none. `cameras` are the model cameras that are the robot's own."""
        try:
            spec = mujoco.MjSpec.from_file(str(model_path))
            if model_path == PANDA:
                # MJCF cannot add a child to a body that an included file defines, so the Panda scene's wrist
                # camera is added here: 6 cm beside the hand, looking along the fingers, tilted 25 degrees toward
                # them.
                spec.body("hand").add_camera(
                    name="wrist", pos=[0.06, 0, 0.02], quat=[0.15305, 0.69035, 0.69035, 0.15305], fovy=58
                )
            self._model = spec.compile()
        except ValueError as error:
            raise StartupError(f"cannot load the MuJoCo model {model_path}: {error}") from None
        name = self._model.names.split(b"\0", 1)[0].decode()  # <mujoco model="...">
        # Hashed as compiled, so that a change to any included file or mesh changes the revision.
        compiled = np.zeros(mujoco.mj_sizeModel(self._model), np.uint8)
        mujoco.mj_saveModel(self._model, None, compiled)
        self._revisions = {"robot_description": name, "model_sha256": hashlib.sha256(compiled).hexdigest()[:12]}
        self._notice = (
            f"MuJoCo simulation of the model {name!r} in {model_path.name}. "
            "Results are simulated physics, not a physical robot."
        )
        if not arm:
            raise StartupError("the mujoco backend needs at least one arm joint (--backend-arg arm=NAME,...)")
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
                    f"the MuJoCo model has no camera named {camera!r}; it has {named}. Name the robot's own cameras "
                    "with --backend-arg cameras=NAME,... (empty for none)"
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

    def apply(self, activity_id: str, contract: Contract) -> None:
        group = self._groups[contract.resources[0]]
        action, limits = contract.action, contract.limits
        judge: TrajectoryJudge | WidthJudge
        if isinstance(action, JointTrajectoryGoal):
            columns, judge = [group.spec.joints.index(name) for name in action.joints], TrajectoryJudge(action, limits)
        else:
            columns, judge = list(range(len(group.spec.joints))), WidthJudge(action, limits)
        group.motion = _Motion(activity_id, self._data.time, columns, judge)

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
                feedback.activities[group.motion.activity_id] = group.motion.status
                if group.motion.status.phase == "done":
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
        if motion is not None and motion.status.phase in ("executing", "finishing"):
            judge = motion.judge
            if isinstance(judge, TrajectoryJudge):
                # joints.trajectory/1: follow the validated path.
                elapsed = data.time - motion.started
                reference, wanted = sample_trajectory(judge.action.points, elapsed)
                columns = motion.columns
                group.reference[columns] = reference
                motion.status = judge.update(elapsed, positions[columns].tolist(), velocities[columns].tolist(), step)
                if motion.status.phase != "failed":
                    rates[columns] = wanted
            else:
                # gripper.width/1: walk every finger's reference toward an equal share of the gap.
                fingers = len(group.reference)
                share = group.low + (judge.action.width_m - group.low.sum()) / fingers
                reach = judge.limits.speed_m_s / fingers * step
                group.reference += np.clip(share - group.reference, -reach, reach)
                motion.status = judge.update(float(positions.sum()), step)
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
