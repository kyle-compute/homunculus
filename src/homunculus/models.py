"""Public request/result types and profile schemas for protocol v1.

The CLI, Python client, HTTP service, and runtime all use these definitions;
JSON Schema and OpenAPI are generated from them. Wire units are metres,
radians, and seconds. Clock readings are decimal-string nanoseconds
(``*_ns``); durations are float seconds. Unless a field says otherwise, a
clock reading on a session, goal, stream, operation, or event comes from the
runtime's ``host_monotonic`` clock, whose epoch is the ``boot_id``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL = "v1"

# Stable error codes and the HTTP status each one travels with. Clients branch
# on the code; the message is for people.
ERROR_STATUS: dict[str, int] = {
    # request shape and access
    "invalid_request": 422,
    "request_too_large": 413,
    "unauthenticated": 401,
    "permission_denied": 403,
    "unknown_robot": 404,
    "not_found": 404,
    "observation_expired": 410,
    "image_expired": 410,
    "no_new_observation": 409,
    # capability and contract
    "unsupported_profile": 422,
    "unsupported_mode": 422,
    "unknown_target": 422,
    "unsupported_parameter": 422,
    "unsupported_sensor": 422,
    "limit_exceeded": 422,
    "unsupported_operation": 501,
    "profile_unavailable": 409,
    # authority and robot state
    "stale_context": 409,
    "session_invalid": 409,
    "session_conflict": 409,
    "resource_busy": 409,
    "resource_faulted": 409,
    "request_conflict": 409,
    "start_condition_failed": 409,
    # target streams
    "stream_closed": 409,
    "sequence_rejected": 409,
    "sequence_conflict": 409,
    "target_expired": 409,
    "basis_invalid": 409,
    # service
    "journal_full": 507,
    "too_many_subscribers": 503,
    "backend_error": 502,
    "internal_error": 500,
}


class RobotError(Exception):
    """A request the service refused, with a stable code and structured details."""

    def __init__(self, code: str, message: str, **details: Any):
        if code not in ERROR_STATUS:
            raise ValueError(f"unregistered error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def info(self) -> ErrorInfo:
        return ErrorInfo(code=self.code, message=self.message, details=self.details)


class Model(BaseModel):
    """Strict base: unknown fields, lossy coercions, NaN, and infinity are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ErrorInfo(Model):
    code: str
    message: str
    details: dict[str, Any] = {}


# --------------------------------------------------------------------------
# Motion profiles: one action type per profile. The profile name carries its
# major version; a different version is a different, unsupported profile.
# --------------------------------------------------------------------------


class BaseVelocityTarget(Model):
    """A planar velocity setpoint at the declared body-fixed control point."""

    frame_id: str = "base_heading"
    forward_m_s: float
    left_m_s: float
    yaw_rate_rad_s: float


class BaseVelocityGoal(BaseVelocityTarget):
    """Timed travel at one velocity. The goal includes stopping afterwards."""

    profile: Literal["base.planar_velocity/1"]
    target: str
    duration_s: float = Field(gt=0)


class TrajectoryPoint(Model):
    time_s: float = Field(ge=0)
    positions: list[float]
    velocities: list[float]


class JointTrajectoryGoal(Model):
    """Named-joint waypoints joined by cubic Hermite segments. Time zero is activation."""

    profile: Literal["joints.trajectory/1"]
    target: str
    joints: list[str] = Field(min_length=1)
    points: list[TrajectoryPoint] = Field(min_length=1)
    # A client may tighten these; omitted values resolve to the binding's.
    path_tolerance: float | None = Field(default=None, gt=0)
    goal_tolerance: float | None = Field(default=None, gt=0)
    after: Literal["hold"] = "hold"


class HandPostureGoal(Model):
    """Move a dexterous hand to a posture the binding advertises by name."""

    profile: Literal["hand.posture/1"]
    target: str
    posture: str
    after: Literal["hold"] = "hold"


class GripperWidthGoal(Model):
    """Move a parallel-jaw gripper toward an absolute gap."""

    profile: Literal["gripper.width/1"]
    target: str
    width_m: float = Field(ge=0)
    complete_on: Literal["width_reached", "contact_detected"] = "width_reached"
    contact_rule: str | None = None
    force_rule: str | None = None
    after: Literal["hold"] = "hold"


class ToolPoseGoal(Model):
    """Move a declared tool origin along a straight segment to a pose, turning by the shortest rotation.

    The pose is in a fixed frame the binding names, so it means the same at observation and at activation.
    Tolerances, if given, replace the binding's defaults within its declared maxima.
    """

    profile: Literal["tool.pose.linear/1"]
    target: str
    frame_id: str = "world"
    position_m: list[float] = Field(min_length=3, max_length=3)
    orientation_xyzw: list[float] = Field(min_length=4, max_length=4)
    speed_m_s: float | None = Field(default=None, gt=0)
    position_tolerance_m: float | None = Field(default=None, gt=0)
    orientation_tolerance_rad: float | None = Field(default=None, gt=0)
    after: Literal["hold"] = "hold"


Action = Annotated[
    BaseVelocityGoal | JointTrajectoryGoal | HandPostureGoal | GripperWidthGoal | ToolPoseGoal,
    Field(discriminator="profile"),
]


# --------------------------------------------------------------------------
# What a backend binding declares for each profile: its supported domain,
# tolerances, and stop thresholds. The accepted contract records these values.
# --------------------------------------------------------------------------


class BaseVelocityLimits(Model):
    profile: Literal["base.planar_velocity/1"]
    # The body-fixed point whose velocity is commanded and measured, and the frames a command may use.
    control_point: str
    frames: list[str]
    max_forward_m_s: float
    max_left_m_s: float
    max_yaw_rate_rad_s: float
    max_goal_duration_s: float
    # The range a stream may choose its target validity from. None: this binding has no stream form.
    min_update_validity_wall_s: float | None = None
    max_update_validity_wall_s: float | None = None
    # Tracking. After transient_s, the measured velocity (averaged over tracking_window_s; 0 means each
    # control period on its own) must stay within tolerance of the command on every axis. The tolerance
    # is the absolute error below or, where larger, max_tracking_error_ratio of that axis's command. An
    # error lasting longer than max_tracking_error_duration_s fails the goal or stream. A timed goal
    # succeeds only after travel was measured within tolerance, so it must outlast the transient and
    # command more than the absolute error on some axis.
    transient_s: float
    max_tracking_error_m_s: float
    max_tracking_error_rad_s: float
    max_tracking_error_ratio: float = Field(0.0, ge=0, lt=1)
    max_tracking_error_duration_s: float
    tracking_window_s: float = 0.0
    # Measured speeds below these, held for settle_time_s, confirm a stop.
    stop_speed_m_s: float
    stop_yaw_rate_rad_s: float
    settle_time_s: float


class JointLimit(Model):
    name: str
    unit: Literal["rad", "m"]
    min_position: float
    max_position: float
    max_velocity: float
    max_acceleration: float


class JointTrajectoryLimits(Model):
    profile: Literal["joints.trajectory/1"]
    joints: list[JointLimit]
    max_points: int
    max_duration_s: float
    start_position_tolerance: float
    start_velocity_tolerance: float
    path_tolerance: float
    goal_tolerance: float
    settle_velocity: float
    settle_time_s: float


class HandPostureLimits(Model):
    profile: Literal["hand.posture/1"]
    postures: dict[str, dict[str, float]]
    max_speed_rad_s: float
    goal_tolerance_rad: float
    settle_time_s: float


class GripperWidthLimits(Model):
    profile: Literal["gripper.width/1"]
    min_width_m: float
    max_width_m: float
    speed_m_s: float
    goal_tolerance_m: float
    settle_time_s: float
    complete_on: list[str]
    contact_rules: list[str] = []
    force_rules: list[str] = []


class ToolPoseLimits(Model):
    profile: Literal["tool.pose.linear/1"]
    control_point: str  # the tool origin whose pose is commanded and measured
    frames: list[str]
    # Axis-aligned bounds on the commanded position, in the first frame; a pose outside is refused.
    workspace_min_m: list[float]
    workspace_max_m: list[float]
    min_speed_m_s: float
    max_speed_m_s: float
    default_speed_m_s: float
    max_rotation_rate_rad_s: float
    # The measured tool origin must stay within path_tolerance_m of the reference segment while it moves.
    path_tolerance_m: float
    # Defaults, and the loosest a goal may ask for. Met once within both for settle_time_s.
    position_tolerance_m: float
    max_position_tolerance_m: float
    orientation_tolerance_rad: float
    max_orientation_tolerance_rad: float
    settle_time_s: float
    max_settle_wait_s: float  # after the segment ends; a pose not reached by then fails the goal


BindingLimits = Annotated[
    BaseVelocityLimits | JointTrajectoryLimits | HandPostureLimits | GripperWidthLimits | ToolPoseLimits,
    Field(discriminator="profile"),
]


class ProfileBinding(Model):
    """One profile bound to one concrete target on this robot."""

    profile: str
    target: str
    role: str
    resources: list[str]
    modes: list[Literal["goal", "stream"]]
    controller: str
    assistance: str
    stop_deadline_wall_s: float
    limits: BindingLimits
    # Supported is not the same as available now; the runtime fills these in.
    available: bool = True
    unavailable_reason: str | None = None


# --------------------------------------------------------------------------
# Identity, capabilities, and observations
# --------------------------------------------------------------------------

Condition = Literal["ready", "active", "stopping", "faulted", "recovering"]
Provenance = Literal["sensed", "estimated", "simulation_ground_truth", "fake"]


class BackendInfo(Model):
    kind: str  # fake, mujoco, task, or a name a backend of your own chooses
    # What moves the robot: nothing ("none": a fake sets its joints as commanded), a physics simulator, or the
    # physical world. A result says no more about a physical robot than this allows.
    physics: Literal["none", "simulated", "real"]
    notice: str
    versions: dict[str, str] = {}


class ClockInfo(Model):
    id: str
    epoch: str
    description: str


class SensorInfo(Model):
    id: str
    # A camera is one of the robot's own: observations record it. A spectator_camera is for people watching
    # its live video; observations never include it.
    kind: Literal["joint_state", "base_state", "camera", "spectator_camera", "task_state"]
    provenance: Provenance
    description: str
    # A camera that can be watched live: the frames per second of its video. None: it takes snapshots only.
    video_fps: float | None = Field(default=None, gt=0)


class Retained(Model):
    """What a controller keeps maintaining after its goal or stream ended."""

    kind: Literal["hold", "fallback"]
    source: str | None
    profile: str | None
    target: dict[str, float]


class ResourceStatus(Model):
    """A control group, its controller, and who is using it."""

    id: str
    role: str
    joints: list[str]
    controller: str
    condition: Condition
    activity: str | None = None
    retained: Retained | None = None
    fault: str | None = None


class SessionStatus(Model):
    held: bool
    principal: str | None = None
    expires_in_s: float | None = None


class ServiceLimits(Model):
    """The service's configured bounds. The runtime enforces them and publishes them unchanged."""

    max_request_bytes: int = 65536
    default_session_lifetime_wall_s: float = 300.0
    max_session_lifetime_wall_s: float = 3600.0
    default_max_execution_s: float = 30.0
    default_max_wall_s: float = 60.0
    max_deadline_s: float = 600.0
    observation_retention: int = 256
    observation_wait_max_s: float = 5.0
    image_store_bytes: int = 64 * 1024 * 1024
    event_retention: int = 1024
    max_event_subscribers: int = 8
    max_video_viewers: int = 4
    # A video link must be opened within this time; a video that is playing goes on.
    video_link_lifetime_s: float = 600.0
    operation_retention: int = 256
    journal_quota: int = 10_000


class Unavailable(Model):
    available: Literal[False] = False
    reason: str


class ControlLoop(Model):
    period_s: float
    ticks: int
    worst_interval_s: float
    execution_lag_s: float | None = None


class RobotDescription(Model):
    protocol: str = PROTOCOL
    robot_id: str
    boot_id: str
    context_id: str
    world_id: str | None
    backend: BackendInfo
    revisions: dict[str, str]
    mode: str
    clock_mode: Literal["realtime", "stepped"]
    clocks: list[ClockInfo]
    frames: list[str]
    sensors: list[SensorInfo]
    resources: list[ResourceStatus]
    profiles: list[ProfileBinding]
    concurrent_groups: list[list[str]]
    management: list[str]
    session: SessionStatus
    limits: ServiceLimits
    persistent_channel: Unavailable
    control_loop: ControlLoop


class Sample(Model):
    """One measurement with its own source, clock, and validity."""

    source: str
    provenance: Provenance
    clock_id: str
    clock_epoch: str
    capture_time_ns: str
    valid: bool
    values: dict[str, Any]


class Pose(Model):
    """A position and orientation in a frame that `describe` lists."""

    frame_id: str
    position_m: Annotated[list[float], Field(min_length=3, max_length=3)]
    orientation_xyzw: Annotated[list[float], Field(min_length=4, max_length=4)]


class CameraImage(Model):
    """What a camera recorded, without the bytes, and what it takes to turn a pixel into a direction.

    Every source states images the same way. The image is undistorted: an ideal pinhole with focal lengths
    focal_px (x, y) and principal point principal_px (x, y), in pixels from the top-left corner. camera_pose is
    where the camera was at capture, with optical axes: x right, y down, z forward. A depth image holds float32
    metres along z, pixel for pixel with the rgb image it was captured with.
    """

    kind: Literal["rgb", "depth"]
    media_type: str
    width: int
    height: int
    clock_id: str
    clock_epoch: str
    capture_time_ns: str
    focal_px: Annotated[list[float], Field(min_length=2, max_length=2)]
    principal_px: Annotated[list[float], Field(min_length=2, max_length=2)]
    camera_pose: Pose


class ImageHandle(CameraImage):
    """A recorded image in an observation; the bytes travel separately, at /images/{handle}."""

    handle: str
    sensor: str


class Observation(Model):
    """An immutable snapshot. Grouping samples does not make them simultaneous."""

    observation_id: str
    robot_id: str
    context_id: str
    sequence: int
    issued_time_ns: str
    samples: dict[str, Sample]
    resources: list[ResourceStatus]
    images: list[ImageHandle] = []


class VideoLink(Model):
    """A link that plays one camera live in a browser. Whoever has it can start watching, without a token, until
    it expires; a video that is playing goes on."""

    camera: str
    fps: float
    url: str
    expires_in_s: float


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------


class SessionOpen(Model):
    request_id: str = Field(min_length=1, max_length=128)
    context_id: str
    lifetime_wall_s: float | None = Field(default=None, gt=0)
    # Operator only: revoke the current holder and start its stop/handover.
    takeover: bool = False


class SessionRenew(Model):
    lifetime_wall_s: float | None = Field(default=None, gt=0)


class Session(Model):
    session_id: str
    robot_id: str
    context_id: str
    principal: str
    state: Literal["active", "closed", "expired", "revoked"]
    reason: str | None = None
    lifetime_wall_s: float
    expires_at_ns: str
    expires_in_s: float


# --------------------------------------------------------------------------
# Finite goals
# --------------------------------------------------------------------------

GoalState = Literal["accepted", "executing", "finishing", "stopping", "succeeded", "cancelled", "failed", "interrupted"]
TERMINAL_GOAL_STATES = ("succeeded", "cancelled", "failed", "interrupted")


class StartConditions(Model):
    """Checked at admission and again immediately before dispatch."""

    required_mode: str | None = None
    max_measurement_age_s: float | None = Field(default=None, gt=0)
    # Bound how far the goal's resources may have moved since this observation.
    observation_id: str | None = None
    max_position_change: float | None = Field(default=None, ge=0)


class GoalSpec(Model):
    """A proposed goal: one action with its deadlines and start conditions."""

    action: Action
    max_execution_s: float | None = Field(default=None, gt=0)
    max_wall_s: float | None = Field(default=None, gt=0)
    start_conditions: StartConditions | None = None


class GoalSubmit(GoalSpec):
    request_id: str = Field(min_length=1, max_length=128)
    session_id: str
    context_id: str


class Contract(Model):
    """The accepted command with every setting resolved to a concrete value."""

    action: Action
    resources: list[str]
    limits: BindingLimits
    max_execution_s: float
    max_wall_s: float
    stop_deadline_wall_s: float
    start_conditions: StartConditions | None = None


class GoalResult(Model):
    """Why a goal ended, with the measurements behind that outcome."""

    reason: str
    detail: str | None = None
    measurements: dict[str, Any] = {}
    completion_observation_id: str | None = None
    stop_confirmed: bool | None = None


class Goal(Model):
    goal_id: str
    request_id: str
    robot_id: str
    context_id: str
    session_id: str
    principal: str
    state: GoalState
    contract: Contract
    accepted_time_ns: str
    activated_time_ns: str | None = None
    finished_time_ns: str | None = None
    result: GoalResult | None = None
    retained: Retained | None = None


class Authority(Model):
    """Session and context for cancel, close, and stop. An operator may omit both for stop."""

    session_id: str | None = None
    context_id: str | None = None


class ValidateRequest(Model):
    """Exactly one of the two, stated as it would be submitted."""

    goal: GoalSpec | None = None
    stream: StreamSpec | None = None


class Validation(Model):
    """A check only: nothing is reserved and later acceptance is not guaranteed."""

    valid: Literal[True] = True
    resources: list[str]
    contract: Contract | None = None
    limits: BindingLimits | None = None


# --------------------------------------------------------------------------
# Target streams
# --------------------------------------------------------------------------


class StreamSpec(Model):
    profile: str
    target: str
    update_validity_wall_s: float = Field(gt=0)
    initial_target: BaseVelocityTarget


class StreamOpen(StreamSpec):
    request_id: str = Field(min_length=1, max_length=128)
    session_id: str
    context_id: str
    basis_observation_id: str


class StreamUpdate(Model):
    session_id: str
    context_id: str
    sequence: int = Field(ge=1)
    basis_observation_id: str
    target: BaseVelocityTarget


class TargetAck(Model):
    sequence: int
    basis_observation_id: str
    expires_at_ns: str


class Stream(Model):
    stream_id: str
    request_id: str
    robot_id: str
    context_id: str
    session_id: str
    principal: str
    profile: str
    target: str
    resources: list[str]
    limits: BaseVelocityLimits
    update_validity_wall_s: float
    state: Literal["open", "stopping", "closed"]
    closure: Literal["client_closed", "expired", "failed", "interrupted"] | None = None
    accepted_sequence: int
    applied_sequence: int | None = None
    expires_at_ns: str
    expires_in_s: float
    update_path: str
    persistent_channel: Unavailable
    detail: str | None = None


class StreamUpdateResult(Model):
    ack: TargetAck
    duplicate: bool
    stream: Stream


# --------------------------------------------------------------------------
# Stop, management, events, service
# --------------------------------------------------------------------------


class ManagementRequest(Model):
    request_id: str = Field(min_length=1, max_length=128)
    context_id: str
    operation: str
    parameters: dict[str, Any] = {}


class Operation(Model):
    """An asynchronous stop or management operation and its evidence."""

    operation_id: str
    kind: str
    robot_id: str
    principal: str
    state: Literal["running", "succeeded", "failed"]
    context_id: str
    new_context_id: str | None = None
    resources: dict[str, Condition] = {}
    reason: str | None = None
    detail: str | None = None
    outcome: dict[str, Any] = {}
    requested_time_ns: str
    finished_time_ns: str | None = None


class Event(Model):
    sequence: int
    time_ns: str
    type: str
    data: dict[str, Any] = {}


class RobotSummary(Model):
    robot_id: str
    backend: BackendInfo


class ServiceInfo(Model):
    service: Literal["homunculus"] = "homunculus"
    version: str
    protocol: str = PROTOCOL
    boot_id: str
    robots: list[RobotSummary]


ValidateRequest.model_rebuild()
