# Usage guide

How each part of `homunculus` behaves, in more detail than the [README](../README.md).

- [Targets](#targets)
- [A finite movement](#a-finite-movement)
- [A continuous target](#a-continuous-target)
- [Two robots, one client](#two-robots-one-client)
- [Stopping, recovery, reset](#stopping-recovery-reset)
- [Output for scripts](#output-for-scripts)
- [Units, frames, and clocks](#units-frames-and-clocks)
- [Structured input](#structured-input)
- [MuJoCo example](#mujoco-example)
- [Rigs](#rigs)
- [Camera images](#camera-images)
- [A downloaded benchmark task](#a-downloaded-benchmark-task)
- [Your own robot](#your-own-robot)
- [Python client](#python-client)
- [Running a service yourself](#running-a-service-yourself)
- [Where things are stored](#where-things-are-stored)
- [Limitations](#limitations)

## Targets

`homunculus demo start` prints where the service listens and where its files are, and saves its
robots as targets. The demo's default target is `fake:arm`. Choose another per command with
`-t fake:g1`, for the shell with `HOMUNCULUS_TARGET=fake:g1`, or persistently with
`homunculus use fake:g1`. `homunculus targets` lists them; `homunculus forget NAME` removes one you connected
yourself. When the default target is removed, no other takes its place: the next command asks
you to choose one.

The CLI remembers the session, context, and the last goal, stream, and operation for each
target, so you do not copy IDs between commands.

## A finite movement

A goal succeeds only when the controller measures completion, and its result
says what the robot is left holding:

```console
$ homunculus goal joints joint4=-2.2 joint6=2.2 --duration 1.5 --wait
goal goal-…: succeeded (completed)
  profile       joints.trajectory/1 on 'arm'; resources arm
  deadlines     30 s execution, 60 s wall
  measured      final_error={joint4=0, joint6=0}, peak_tracking_error=0, goal_tolerance=0.02
  evidence      observation obs-…
  left holding  hold at joint1=0, joint2=0, joint3=0, joint4=-2.2, joint5=0, joint6=2.2,
                joint7=-0.7853
```

Without `--wait` the command returns once the goal is accepted; follow it with
`homunculus goal inspect --wait`, stop it alone with `homunculus goal cancel`. Ctrl-C
during a wait ends the wait and nothing else: the command says that the goal
goes on and how to end it.

A goal has two deadlines, 30 s of execution time and 60 s of wall time unless
`--max-execution` and `--max-wall` say otherwise; a goal that passes one is
stopped and fails. A movement that states how long it takes (a trajectory,
timed travel) is refused before anything moves when that time, and the rest
the robot must then be measured at, cannot fit the execution deadline. On a
robot that runs in real time the wall deadline is compared the same way.

`--dry-run` checks a request as it would be submitted, its deadlines and start
conditions included, without reserving or moving anything, and
`--request-id ID` makes a retry return the first execution instead of running a
second one. `homunculus manage` and `homunculus session open` take `--request-id` too.
An ID names one request, its session included. The same ID with other
arguments, or sent from a later session, is refused with `request_conflict`;
the refusal names the goal the ID belongs to, and `homunculus goal inspect` with
that goal ID shows what it did.

An empty ID or target name, as a shell variable that is not set leaves, is an
error with exit status 2, and nothing is sent. It never stands for a new
request, the last goal, stream, or operation, or the default target:

```console
$ homunculus goal cancel "$GOAL"
usage: homunculus goal cancel [-h] [-t TARGET] [--json] [--no-wait] [--timeout S]
                         [GOAL_ID]
homunculus goal cancel: error: argument GOAL_ID: it is empty; if a shell variable should hold it, that variable is not set
```

## A continuous target

```sh
homunculus use fake:g1 && homunculus session open
homunculus stream open --forward 0.3 --validity 1    # each target is valid for 1 s
homunculus stream update --forward 0.2 --yaw 0.4     # replaces the target; no stop in between
homunculus stream inspect                            # accepted and applied sequence, time to expiry
homunculus stream close                              # or let it expire: the base stops either way
```

Run these lines together, pasted or from a script. A target is valid for at
most 1 s on this robot, so when the commands are typed one by one the stream
has expired, and stopped the base, before the update arrives. The update is
then refused with `stream_closed`, and `homunculus stream inspect` shows the stream
`closed (expired)`.

Each target carries the next sequence number and is based on an observation.
It stays valid for the stream's window counted from when that observation was
issued, not from when the target arrives: the time spent deciding and sending
comes out of the window. A late, duplicate, or reordered update cannot extend
or reopen a stream.

Without `--basis` the CLI reads an observation as it sends the target, so the
window starts then. A client that decided on an earlier observation names it
and has the rest of that observation's window:

```sh
SEEN=$(homunculus --json observe | python3 -c 'import json, sys; print(json.load(sys.stdin)["observation_id"])')
homunculus stream update --forward 0.2 --basis "$SEEN"
```

An observation older than the window cannot open a stream (`target_expired`).
An update cannot be based on an observation older than the one before it, or
on one from another execution context (`basis_invalid`).

Closing a stream that has already expired shows it closed and exits with 0.
The exit status is 1 only when the stop was not confirmed.

A client too slow to stream can ask for timed travel instead. The goal includes
the stop, and it is judged against measured velocity, not elapsed time:

```sh
homunculus goal base --forward 0.3 --duration 2 --wait
```

A base that does not follow the command fails the goal with `tracking_error`
instead of succeeding when its time is up. Travel that could not be told apart
from standing still is refused before anything moves; `homunculus describe` shows
the shortest duration and the slowest speed each base can verify.

## Two robots, one client

`fake:arm` offers joint trajectories and a gripper; `fake:g1` offers base
velocity and named hand postures. The client reads that from `describe` and
refuses what a robot does not offer before sending anything:

```console
$ homunculus -t fake:g1 goal joints joint4=-2.2
error: fake-g1 does not offer joints.trajectory/1 as a goal; it offers: base.planar_velocity/1 on 'base' (goal/stream), hand.posture/1 on 'left_hand' (goal) [unsupported_profile]
```

The service applies the same rule to requests that reach it some other way.

## Stopping, recovery, reset

```sh
homunculus stop                 # stop everything this session commands; waits for measured standstill
homunculus stop --operator      # needs no session and no current context
homunculus manage recover       # after an unconfirmed stop faulted a resource
homunculus manage world.reset   # fake and mujoco: new world, new context; open a new session afterwards
homunculus events               # numbered runtime events; --follow to keep reading
```

The service answers a stop, a goal cancellation, and the closing of a stream or
session from workers it keeps for them. Other requests do not delay them,
however many are waiting, for example on cameras.

An operator can take the session from its holder with
`homunculus session open --takeover`. The holder is revoked at once and its motion
is stopped; while that stop is still being confirmed the command is refused
with `resource_busy`, and it succeeds when repeated. The new session belongs to
the operator, so the target acts as operator until that session is closed.

A session that was closed, revoked, or has expired ends nothing afterwards,
whoever held it. `homunculus stop` in such a session is refused and points to
`homunculus stop --operator`, which needs no session.

Events are numbered from 1 in each run of the service. A `--follow` reader that
fell behind, or that reconnects after the service restarted, gets an
`events.dropped` line before the events that follow. Events are for watching;
`homunculus goal inspect` and `homunculus operation inspect` are the record of what
happened. A service serves 8 event readers at a time and refuses a further one
with `too_many_subscribers`.

## Output for scripts

Text is the default. `--json` prints one JSON document on standard output
(`homunculus events` prints one per line). Errors go to standard error; with
`--json` an error is one document too,
`{"error": {"code": …, "message": …, "hint": …, "details": …}}`. Service logs go
to the service's own log file.

| Exit status | Meaning |
| --- | --- |
| 0 | Done, or accepted and still running. |
| 1 | The robot's outcome was a failure, a cancellation, or the wait timed out. |
| 2 | The command line or an input file is not usable. |
| 3 | The service refused the request; the message ends with a stable code such as `[stale_context]` (the `code` field with `--json`). |
| 4 | Nothing could be sent, started, or stopped: no such target, no recorded session, the service is unreachable, a service already uses those files (`demo start`, `serve`), or one holds them without answering (`demo stop`). Also when the CLI cannot read or write a file it needs (its own files, or the `--save` directory); the message names it. |

A command interrupted with Ctrl-C exits with 130, and one whose reader closed
its output (`homunculus events --follow | head -3`) with 141 the next time it has a
line to write. Neither prints an error, with two exceptions, because the robot
or the service goes on without the command. One that was waiting for a goal,
an operation, or a closing stream says that only the wait ended. One whose
request the service had not answered says so: a request that changes something
may still have been received, and for a goal, a management operation, or a
session open the message names the request ID to repeat it under.

## Units, frames, and clocks

Values are in metres, radians, seconds, newtons, and newton-metres. A rotating
joint is measured in radians and a sliding joint in metres. `homunculus describe`
lists the unit of every joint that takes positions, the frames the robot
knows, and the limits of each profile and of the service. A base velocity is
given along the base's own heading (frame `base_heading`); a base position is
reported in `world`.

In JSON, a name that ends in a unit states it: `expires_in_s` is in seconds,
`width_m` in metres, `forward_m_s` in metres per second, `yaw_rate_rad_s` in
radians per second. Values listed per joint follow the joint: radians, radians
per second, and newton-metres for a rotating joint; metres, metres per second,
and newtons for a sliding one. The tolerances and errors of a joint movement
are in the joint's position unit.

A field ending in `_ns` is a clock reading in nanoseconds, written as a decimal
string. Every measurement names the clock it was read on (`clock_id`) and that
clock's epoch (`clock_epoch`); `homunculus --json describe` lists the clocks.
Readings from different clocks, or from one clock in different epochs (a
restart or a world reset starts a new one), cannot be compared.

## Structured input

The flags cover common requests. Anything else goes in a JSON file (or `-` for
standard input), either a bare action or a request with deadlines and start
conditions:

```sh
cat > move.json <<'JSON'
{
  "action": {
    "profile": "joints.trajectory/1", "target": "arm", "joints": ["joint4", "joint6"],
    "points": [
      {"time_s": 1.0, "positions": [-1.9, 1.9], "velocities": [-0.3, 0.3]},
      {"time_s": 2.0, "positions": [-2.2, 2.2], "velocities": [0.0, 0.0]}
    ]
  },
  "max_execution_s": 5,
  "start_conditions": {"max_measurement_age_s": 0.5}
}
JSON
homunculus goal submit --file move.json --wait
```

A request file is JSON in UTF-8; a byte-order mark at its start is accepted.
It is read strictly. A field that a request does not have is an error (exit
status 2), and so is a value given both in the file and as a flag.
Leave `session_id` and `context_id` out and the CLI fills in the ones it
remembers; when the file states them, they are sent as written.

`homunculus stream open --file` takes a stream request: profile, target, validity,
and first target, and optionally its request ID, session, context, and basis
observation. `homunculus stream update --file` takes a bare target, or an update
request that also states its session, context, sequence number, or basis
observation (here `$SEEN`, read as shown under *A continuous target*):

```sh
homunculus stream update --file - <<JSON
{
  "basis_observation_id": "$SEEN",
  "target": {"forward_m_s": 0.1, "left_m_s": 0.0, "yaw_rate_rad_s": 0.2}
}
JSON
```

What a stream file leaves out the CLI fills in, as it does for the flags. The
file replaces the velocity flags and `--basis`, so one given beside it is
refused.

## MuJoCo example

```sh
homunculus demo start panda                 # needs the [mujoco] extra; downloads the Panda the first time
homunculus session open
homunculus goal joints joint2=0.6 joint4=-1.8 --duration 2 --wait
homunculus goal gripper 0.02 --wait
homunculus observe                          # measured joint positions, velocities, actuator forces
homunculus observe --camera wrist --camera workspace   # and what the robot's cameras see
homunculus demo stop
```

`panda` is one of the [rigs](#rigs) that ship with homunculus. Its robot is the
Franka Emika Panda from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
unchanged (commit `370c439`, Apache-2.0): its
meshes, masses, joint ranges, and force-limited position actuators. It stands on
a table, with a mug and a bowl from Google Scanned Objects in front of it; both
are free to move. The arm
joints are `joint1` to `joint7`. The two fingers share one actuator, which pulls
a tendon over both, and an equality constraint keeps them level. Nothing
compensates gravity, so a held pose sags: after the first goal above, `joint2`
holds 42 N·m and stops 0.009 rad short, inside the 0.02 rad goal tolerance.

Success is judged from the simulated joint state. A move the arm cannot follow
fails instead of succeeding when its time is up:

```console
$ homunculus goal joints joint2=1.6 joint4=-0.8 --duration 2 --wait    # the hand meets the table
goal goal-…: failed (path_tolerance_violated)
  profile       joints.trajectory/1 on 'arm'; resources arm
  deadlines     30 s execution, 60 s wall
  detail        measured position left the tolerance around the reference path
  measured      joint=joint2, tracking_error=0.1001, path_tolerance=0.1, at_s=0.978
  stop          confirmed by measurement
  left holding  fallback at joint1=…, joint2=0.9889, joint3=…, joint4=-1.306, joint5=…,
                joint6=1.247, joint7=-0.7853
```

That is the output after the commands above, on MuJoCo 4.0.1; the numbers
depend on the pose the move starts from and on the MuJoCo version. MuJoCo's
contact list has the hand on the table top when the goal fails.

It has no stream profile. It has two cameras of its own, as a real Panda
set up for manipulation would:

- `wrist`, on the hand 6 cm beside it, looking along the fingers and tilted 25°
  toward them;
- `workspace`, fixed 1.3 m in front of the robot's base and 0.85 m up, aimed at
  the table 0.45 m in front of the base.

`homunculus observe --camera wrist` saves a 640x480 JPEG under `$HOMUNCULUS_HOME/images`
(or `--save DIR`) and prints its path, the lens, and where the camera was (see
[Camera images](#camera-images)). A third camera, `overview`, looks at the
scene from the front and to the side. It is for people: `homunculus video overview`
plays it live (see [Live video](#live-video)), and an observation refuses it.

## Rigs

The MuJoCo backend serves a rig: a folder that holds a `rig.toml` and the MJCF
model it names. A rig is data. Nothing in it runs. Three ship with homunculus,
and `homunculus rigs` lists them:

| Rig | Robot, from MuJoCo Menagerie | Scene | The robot's cameras |
| --- | --- | --- | --- |
| `panda` | `franka_emika_panda`: 7 arm joints and a two-finger gripper | a table with a mug and a bowl | `wrist`, `workspace` |
| `so101` | `robotstudio_so101`: 5 arm joints and a jaw, driven as a sixth joint | a desk with a figure and a plate | `wrist_cam` |
| `ur5e` | `universal_robots_ur5e`: 6 arm joints, no gripper | a bench with a mug, a plate and a toy bee | `realsense`, fixed on a pole |

Each rig also has an `overview` camera, for people watching.

```sh
homunculus install so101       # download its robot, then check that the rig works here
homunculus demo start so101    # short for: --backend mujoco --backend-arg rig=so101
homunculus serve ./my-arm      # a rig folder of your own, named by its path
```

A bare word names a shipped rig, in any case. A path has a slash or starts with
a dot. The robot is named after the rig's folder, so the `so101` rig's target is
`so101`; `--backend-arg robot_id=NAME` gives it another name.

### The rig file

`rig.toml` says what MJCF does not. This is the Panda's:

```toml
description = "Franka Emika Panda at a table, with a wrist camera and a workspace camera"
menagerie = "franka_emika_panda"   # optional: the Menagerie robot whose files load beside this folder's
model = "table.xml"                # an MJCF file in this folder
arm = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
gripper = ["finger_joint1", "finger_joint2"]   # optional: finger joints that slide
cameras = ["wrist", "workspace"]   # optional: the model cameras that are the robot's own

[[mount]]                          # optional, repeatable: a camera fixed to one of the robot's bodies
camera = "wrist"
body = "hand"
pos = [0.06, 0.0, 0.02]            # metres, in the body's frame
quat = [0.15305, 0.69035, 0.69035, 0.15305]   # w x y z; the camera looks along its -z, with y up
fovy = 58.0                        # vertical field of view, in degrees
```

A key that is not one of these is refused. With `menagerie`, the robot's files
load as if they were in the rig's folder, so the model includes them by name:
`<include file="panda.xml"/>`. A file in the rig's folder cannot have the name
of one of the robot's files. Without `menagerie`, the folder holds every file
the model needs. `[[mount]]` is there because MJCF cannot add a camera to a body
that an included file defines. A name that starts with a dot, such as `.git`,
is not part of a rig.

The model must meet three conditions, and the service refuses to start with the
reason when it does not:

- Each arm joint is a hinge or slide joint with a range; each finger joint is a
  slide joint with a range. The gripper's width is the sum of the finger
  positions.
- Each of those joints is moved by exactly one position actuator: fixed gain
  and affine bias, as `<position kp=… kv=…>` builds and Menagerie's `<general>`
  actuators are. The actuator acts on the joint, or on a fixed tendon over
  several joints, as the Panda's finger actuator does.
- The model has a keyframe named `home`, where the world starts and resets.

Each camera that `cameras` names is one of the robot's own; every other named
camera is a spectator camera, for live video only. Cameras are rendered at the
model's offscreen size (`<global offwidth=… offheight=…/>` under `<visual>`;
640x480 unless the model sets it) and must be set by `fovy`; a camera with a
`sensorsize` is refused. Rendering needs OpenGL. On a machine without a
display, set MuJoCo's `MUJOCO_GL` environment variable to `egl` or `osmesa`.
On Linux x86_64 with Mesa's software renderer, `egl` ran every rig and the
tests, and `osmesa` ran the SO-101 and gave the same picture. A camera that
cannot render is refused with that advice.

### Where the robot comes from

homunculus has no download, checksum or cache code. The
[`mujoco-menagerie`](https://pypi.org/project/mujoco-menagerie/) package does
that work, and homunculus's `mujoco` extra pins it to one version, 2026.11.3.
That version's robots are Menagerie at commit `370c439`. The package downloads
one archive per robot, checks it against a sha256 it carries, and unpacks it
into its own cache folder (`~/Library/Caches/mujoco_menagerie` on macOS;
`MENAGERIE_CACHE_DIR` moves it).

| Robot | Download | On disk | License |
| --- | ---: | ---: | --- |
| `franka_emika_panda` | 2.9 MB | 32.8 MB | Apache-2.0 |
| `robotstudio_so101` | 3.9 MB | 16.6 MB | Apache-2.0 |
| `universal_robots_ur5e` | 2.5 MB | 29.6 MB | BSD-3-Clause |

A rig's robot is downloaded the first time the rig is installed or started.
After that, the rig needs no network. With downloads blocked, an installed
`so101` started and moved; `homunculus install ur5e`, whose robot was not
downloaded, exited with status 1 and said why.

`homunculus --json describe` reports what a rig was built from, under
`revisions`:

```json
{
  "robot_description": "SO-101 at a desk",
  "model_sha256": "672371e73152",
  "menagerie": "robotstudio_so101@8b739c94709a"
}
```

`menagerie` is the robot and the start of its git tree id in Menagerie.
`model_sha256` is over every file the model can include, the rig's and the
robot's, by name and content. Neither depends on where the files are or on the
machine, so two machines that serve the same rig report the same values.

### What install checks

`homunculus install RIG` loads the rig and drives it through the runtime, as a
client would, on a clock that it steps itself:

1. The rig holds its start pose. After 2 s, no arm joint has moved more than
   the goal tolerance, 0.02 rad or m.
2. Each arm joint in turn moves toward the middle of its range and back. It
   moves 0.15 rad or m, or a fifth of its range if that is less. Both goals
   must succeed, judged from measurement like any goal.
3. The gripper, if the rig has one, closes halfway and opens.
4. Each of the robot's cameras takes a picture, saved in
   `$HOMUNCULUS_HOME/images` or in `--save DIR`.

The command exits with status 0 when every step passes and 1 when one fails.
`--json` gives each step. A rig that fails step 1 is not driven further.

A pass shows that the rig is wired correctly: the names in `rig.toml` exist, the
actuators hold and move their joints, and the cameras render. It does not show
that the arm is strong enough for a task, or that a camera is aimed well. In a
trial of these steps on the SO-101, an actuator weakened to 3% of its stiffness
still passed, and so did a wrist camera moved inside the gripper. Look at the
pictures.

The tests plant these faults:

| Fault | Where it is caught |
| --- | --- |
| A joint, a camera or a mount body that the model does not have | when the rig loads, with the reason |
| A Menagerie robot that does not exist; a key that `rig.toml` does not define | when the rig loads, with the reason |
| A joint that gravity pulls 0.059 rad from its start pose | step 1 |
| A post in the way of a joint | step 2 |

### Menagerie's arms

[`examples/menagerie_sweep.py`](../examples/menagerie_sweep.py) measures which
of Menagerie's 28 arms the backend drives as they are. For each arm it writes a
rig by rule and runs the same check. The rule: Menagerie's own scene, and as arm
joints every hinge or slide joint with a range that one actuator drives. With
MuJoCo 4.0.1, 17 pass: `agilex_piper`, `fanuc_crx5ia`, `flexiv_rizon4`,
`franka_emika_panda`, `franka_fr3`, `franka_fr3_v2`, `i2rt_yam`, `kinova_gen3`,
`low_cost_robot_arm`, `robotis_omy_3m`, `seeed_rebot_devarm`, `trs_so_arm100`,
`ufactory_lite6`, `ufactory_xarm7`, `unitree_z1`, `universal_robots_ur10e` and
`universal_robots_ur5e`. The other 11:

| Arms | Result | Why |
| --- | --- | --- |
| `aloha`, `dynamixel_2r`, `robotstudio_so101`, `trossen_wxai` | do not load | The model has no keyframe named `home`. The `so101` rig's `desk.xml` adds one |
| `arx_l5` | does not load | Its camera sets a sensor size |
| `flexiv_rizon4s`, `franka_fr3_duo`, `kuka_iiwa_14`, `rethink_robotics_sawyer`, `trossen_vx300s`, `trossen_wx250s` | fail step 1 | A joint sags 0.0205 to 0.0713 rad in 2 s, because the backend does not compensate gravity |

A pass says that the backend can drive those joints. It does not say that a
finished rig exists for that arm: that takes a scene, cameras, and a look at
what the cameras see. `kinova_gen3` passes on 3 of its 7 joints; the other four
have no range, and the backend needs one. The sweep downloads 63 MB the first
time.

## Camera images

`homunculus observe --camera NAME` records one of the robot's own cameras, the
ones `homunculus describe` lists under `cameras`, and saves each image. The files go
to `$HOMUNCULUS_HOME/images` (`~/.homunculus/images` by default), or to
`--save DIR`. The command prints each file's absolute path, so an agent that
can open image files sees the picture next. With `--json`, each entry of
`images` has a `file` field with the path.

Every image says how it was taken, the same way whatever produced it:

| Field | Meaning |
| --- | --- |
| `kind`, `media_type`, `width`, `height` | `rgb` or `depth`, the encoding (`image/jpeg`, `image/png`, or `application/x-npy` for depth), and the size in pixels |
| `capture_time_ns`, `clock_id`, `clock_epoch` | When the camera captured it, on which clock |
| `focal_px`, `principal_px` | The lens, as an ideal pinhole: focal lengths and principal point in pixels, from the top-left corner. Images are undistorted. |
| `camera_pose` | Where the camera was at capture: `position_m` and `orientation_xyzw` in `frame_id`, with optical axes x right, y down, z forward |

A depth image holds float32 metres along the optical axis, pixel for pixel
with the `rgb` image captured with it. A pixel (u, v) with depth d is the point
((u + 0.5 − cx)·d/fx, (v + 0.5 − cy)·d/fy, d) in the camera's optical frame;
`camera_pose` takes it to the world. The MuJoCo backend sends `rgb` only; the
task backend sends both.

A camera that `homunculus describe` lists under `spectators` is for people: live
video plays it, and an observation refuses it with the robot's own cameras.

## Live video

A camera that `homunculus describe` lists with live video, the robot's or a
spectator's, can be watched in a browser. `homunculus video` prints a link to it:

```console
$ homunculus video overview
camera overview, live at 20 frames a second:
  http://127.0.0.1:…/v1/robots/panda/video?camera=overview&until=…&signature=…
Open it in a browser within 600 s. Anyone who has the link can watch, without a token.
```

A browser cannot send the token, so the link carries a signature instead. The
signature admits a browser to that one camera of that robot, until the time in
the link. A video that is already playing goes on after that time. A restart of
the service ends every link; `homunculus video` makes a new one. With `--json` the
command prints `camera`, `fps`, `url`, and `expires_in_s`.

The video is one HTTP response whose parts are images
(`multipart/x-mixed-replace`), and the browser shows each part as it arrives.
No page or player is needed. Each viewer gets frames captured for it alone, and
they are not recorded, unlike the images of `homunculus observe`. The Panda's
cameras send 640x480 JPEG frames (quality 90). Watching `overview` at the table
scene: 44.1 kB a frame and 0.87 MB/s at 19.7 frames a second, with one viewer
and with each of four.

The service plays at most 4 videos at once; `homunculus describe` lists the limit.
A link that cannot play shows the reason in the browser in place of a video:
the link expired, the service restarted since it was made, 4 videos are
already playing (`too_many_subscribers`), or the camera failed to capture. An
error after the video has begun ends the video, and the service log says why.

## A downloaded benchmark task

The `task` backend connects one existing task (tested with
`g1-dining-salt-mill`) without copying or changing it. It starts the task's own
simulator entry point with an interpreter you provide, drives it through the
task's own HTTP controls, and leaves the recorded episode where the task's own
verifier can grade it. The task and its assets stay where they are.

```sh
TASK=/path/to/g1-dining-salt-mill

# The task's own pinned environments. Nothing is installed into homunculus.
uv venv --python 3.11 task-env
uv pip install --python task-env/bin/python --require-hashes -r $TASK/environment/requirements.txt
uv venv --python 3.12 verifier-env
uv pip install --python verifier-env/bin/python --require-hashes -r $TASK/tests/requirements.txt

# Observe and act through the CLI. Every physics step is recorded in state_dir.
SCENE_SEED=20261025 homunculus demo start --backend task --backend-arg task_dir=$TASK \
    --backend-arg python=task-env/bin/python --backend-arg state_dir=episode-1
homunculus session open
homunculus observe --camera head --save images     # RGB as PNG, metric depth as .npy
homunculus goal base --forward 0.2 --duration 2 --wait
homunculus goal gripper 0.03 --wait
homunculus manage episode.finish                   # ends the episode the way the task defines it
homunculus demo stop

# Seal and grade the episode with the task's unchanged collect() and grade().
task-env/bin/python -I -B -c '
import importlib.util, sys
spec = importlib.util.spec_from_file_location("task_collect", sys.argv[1] + "/environment/simulator/collect.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.collect(sys.argv[1] + "/environment/simulator", sys.argv[2])' $TASK episode-1
verifier-env/bin/python -I -B -c '
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("task_grade", sys.argv[1] + "/tests/grade.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
state = Path(sys.argv[2])
print(json.dumps(module.grade(state, state / "scene/scene.xml"), indent=2))' $TASK episode-1
```

The two functions are called directly because their command-line entry points
hard-code the paths of the task's containers (`/opt/simulator`, `/state`,
`/logs/verifier`). `collect()` also checks every simulator source file against
the task's `frozen-runtime.json`, so it fails if the task was altered.

To run the task's own containers with the CLI as the agent's only control, the
service attaches to a simulator that is already running instead of starting one:

```sh
homunculus serve --backend task --backend-arg url=http://127.0.0.1:8080 \
    --backend-arg name=g1-dining-salt-mill --port 8090 --home /robot-service
```

That simulator records the episode in its own state directory, where the
task's `collect` and verifier find it as usual.

What the adapter maps, and what it does not:

- `base.planar_velocity/1` as a timed goal, executed as the task's `step`
  actions and finished by a measured stable stand. Travel is verified against
  the base velocity measured over one 0.75 s gait cycle: after the first
  second, an error above 70 % of a commanded component (at least 0.05 m/s or
  0.12 rad/s) that lasts longer than 0.75 s fails the goal with
  `tracking_error`, so walking into the table fails instead of succeeding when
  its time is up. The task's learned walking policy follows velocity loosely,
  which is why the tolerance is that wide; judge distance by the measured
  displacement in the result, reported in the world frame (the command is in
  the base's heading frame).
- `gripper.width/1` on the left parallel gripper, executed as the task's `grip`.
  The task is then stepped at rest until the measured gap has stayed within
  3 mm of the request for 0.2 s and the jaws are still, for at most 4 s; a gap
  that does not get there fails with `width_not_reached`.
- `tool.pose.linear/1` on `left_hand` and `right_hand` (the task's `left_tool`
  and `right_tool` sites), executed as the task's `step` actions with a hand
  target moving every 0.1 s along a straight segment and a shortest-path turn,
  both on a smoothstep. The measured site must stay within 5 cm of the segment
  while it moves (`path_tolerance_violated` otherwise, and the arm then holds
  where it was measured), and must then hold within 1 cm and 0.08 rad of the
  pose for 0.2 s within 3 s (`goal_not_reached` otherwise). A goal may loosen
  those to 5 cm and 0.3 rad, for example while carrying something. Positions
  are bounded as the task bounds hand targets: z from 0.55 to 1.35 m. The
  hand target stays fixed in the world while the base walks.
  On Modal, 46 of 48 free-space goals (each hand; 10 cm moves and 0.5 rad
  turns at 0.04 to 0.2 m/s) succeeded with at most 2.1 cm off the segment and
  8.6 mm and 0.026 rad from the pose; the two failures stopped 2 to 3 cm short
  with the right hand low beside the body.
- `hand.posture/1` on the right hand's articulated fingers, with the postures
  `open` (the task's finger targets at opening 1) and `closed` (the angles an
  empty hand reaches at opening 0, where its fingers stop on each other about
  0.27 rad short of the task's targets), executed as the task's `grip`. Fingers closed on an object stop short of
  `closed`, so that goal fails with `posture_not_reached` and the contact
  force; that is how a grasp shows.
- The four robot cameras, the base, arm, gripper, and hand state, and the
  task's remaining budgets as observations. Object poses are not exposed, as
  in the task's own interface. The cameras take snapshots only: each capture
  uses the task's budget of 300, so they offer no live video.
- Not mapped: the task's planned `reach`, `move_hand` (a pose goal does the
  same motion through `step`, and is measured on the way), `walk_to`, and
  `wait`. Goals run one at a time, although the task could move several parts
  in one step.

The task's world is stepped: physics advances only while a goal runs, so there
is no stream profile, and each goal spends several of the task's 6,000 control
calls. One service runs one episode; start a new one with a new `state_dir`, or a new simulator.
`SCENE_SEED` selects the task's scene (unset: the task picks one at random).

## Your own robot

A backend is a Python class that subclasses `homunculus.runtime.Backend`. The
runtime does the rest: sessions, ownership, deadlines, repeated requests, the
order of stops, restart recovery, the journal, and the HTTP service.
[`examples/lift.py`](../examples/lift.py) is a complete backend, a simulated
one-joint lift, in about 150 lines. Copy it to start your own.

```sh
cd examples
PYTHONPATH=. homunculus demo start --backend lift:robots --backend-arg gain=15
homunculus session open
homunculus goal joints lift=0.5 --duration 2 --wait
homunculus demo stop
```

`--backend MODULE:FUNCTION` names a function that returns robot IDs mapped to
backends. Each `--backend-arg KEY=VALUE` reaches that function as a keyword
argument, as a string; a key it does not take is refused with the ones it
does. The module must be importable: installed, or on `PYTHONPATH`.

| Method | What it does |
| --- | --- |
| `describe()` | Declares resources, the profiles each target takes with its limits and tolerances, sensors, clocks, management operations, and the control period. The runtime refuses whatever is not declared here. |
| `apply(id, contract)` | Starts a goal. `contract.action` is the request with every setting resolved; `contract.limits` is your binding's. |
| `stop(resources)` | The fallback: end motion and hold where the joints are measured to be. |
| `advance()` | Reads, or steps, the robot. Returns one `Sample` per resource with `positions` and `velocities` maps, each running goal's status, which resources are measured at rest, and what each holds. |
| `track`, `capture`, `manage`, `close` | Needed only for what `describe()` advertises: a stream, a camera, a management operation, something to release. |

Rules a backend keeps:

- Success is not the backend's to decide. In `apply()`, make the profile's
  judge from `homunculus.profiles` (`TrajectoryJudge`, `WidthJudge`,
  `PostureJudge`, `VelocityJudge`); in `advance()`, give it what was measured
  and report what it returns. Every backend then succeeds and fails by the same
  rules and reports the same measurements.
- `advance()` returns promptly. A backend that waits on its robot does that in
  a worker thread of its own, as `backends/task.py` does.
- A problem found while starting is a `StartupError` with a message for the
  person who started the service.
- `close()` releases what the backend holds, such as a process or a
  connection. The service calls it when it ends, however it ends, a start that
  failed included. A backend is a context manager, so code of your own builds
  one in a `with`.
- `describe()` says what moves the robot, in its `BackendInfo`: `physics` is
  `none` (a fake that sets its joints as commanded), `simulated`, or `real`.
  `homunculus describe` shows it after the robot's name, as in
  `panda  (mujoco backend, simulated physics)`.
- `capture()` runs on request threads while the control thread calls
  `advance()`, so it reads the robot's state under a lock of its own, as
  `backends/mujoco.py` does. It returns `Frame`s: the bytes, and a
  `CameraImage` in the conventions of [Camera images](#camera-images).
  Convert what your source gives into them there: undistort the image, state
  the lens in pixels, and turn the camera's rotation to optical axes, as
  `backends/task.py` does for the task simulator's MuJoCo camera axes.
- A camera with `kind="camera"` is one of the robot's own; one with
  `kind="spectator_camera"` is for people and plays only as live video. A
  camera whose `SensorInfo` has `video_fps` is captured that many times a
  second for each viewer of its live video, and then returns one `rgb` frame
  in a format browsers show, such as JPEG. Leave `video_fps` out for a camera
  whose captures are costly or counted.

A new kind of motion is a new profile: its goal and limits models in
`models.py` (added to the `Action` and `BindingLimits` unions), its admission
rules in `profiles.resolve_action` (and in `stated_duration_s`, if a request
states how long its motion takes), and its judge in `profiles.py`. The
runtime, the service, and the client stay as they are. `homunculus describe` lists
the new profile's limits as its binding declares them, and `homunculus goal submit
--file` sends its goals. A `homunculus goal` command of its own, and a shorter
layout for its limits in `_limit_rows`, are optional additions to `cli.py`.

## Python client

The CLI is a thin layer over this client; both use the same typed models.

```python
from homunculus.client import RobotClient, TargetStore
from homunculus.models import GripperWidthGoal

store = TargetStore()  # the targets `homunculus connect` / `homunculus demo start` saved
# on_change=store.save records the session and the last goal as they change, as the CLI does
with RobotClient(store.get("fake:arm"), on_change=store.save) as robot:
    print([p.profile for p in robot.describe().profiles])
    with robot.session():  # takes control for this block; see below if the CLI already holds it
        robot.submit_goal(GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=0.03))
        goal = robot.wait_goal()
        print(goal.state, goal.result.measurements, goal.retained)
```

`robot.session()` takes the control session when its block starts and releases
it when the block ends, after an error too, so a script that fails does not
leave the robot held. A robot has one control session. If
`homunculus session open` already holds it for this target, `robot.session()`
is refused with `session_conflict`: leave that `with` out to work in the CLI's
session, or release it first with `homunculus session close`.
`open_session()` and `close_session()` are for a session that outlives one
block, as the CLI's does from one command to the next.

## Running a service yourself

```sh
homunculus serve --backend fake --home ./svc --port 8000    # foreground; Ctrl-C stops it
homunculus connect --home ./svc                             # in another shell: adds its robots as targets
```

The service listens on 127.0.0.1 by default and requires a bearer token for
everything under `/v1`. Two paths need none: `/healthz` (whether the service is up) and
`/openapi.json` (the generated description of the HTTP interface). A live video
takes the signed link that `homunculus video` prints in place of the token (see
[Live video](#live-video)).
It writes two tokens to `<home>/tokens.json` (mode 0600): a client token, and an
operator token for stopping without a session and for management operations.
`<home>/journal.sqlite` keeps request identities and results across restarts:
after a restart, unfinished goals read as `interrupted` and are never replayed.
A retry with the same request ID returns that record when it is sent before a
new session is opened; afterwards it is refused and names the goal to inspect.
For a remote service use `homunculus connect URL --token … [--operator-token …]`.
The client connects straight to the address a target names. It uses no proxy
from the environment (`HTTP_PROXY`, `ALL_PROXY`, the system's proxy settings),
so a robot's commands and tokens are never sent to one.
Connecting again to the same service replaces the saved tokens and keeps what
the CLI remembers for its robots, the open session included. A request body is
limited to 65,536 bytes and must declare its length; a chunked body is refused.

A home belongs to one service at a time. The running service holds the journal
file, so a second `homunculus serve` on the same home exits with status 4 and names
the first, and other programs cannot open `journal.sqlite` until the service
stops. The hold ends with the process, however it ends, so a service that was
killed can be started again on the same home. A service that cannot start for
another reason (the port is taken, the home cannot be written, `journal.sqlite`
or `tokens.json` is damaged, the backend refuses an option or a model) exits
with status 1 and one line saying why.

## Where things are stored

`$HOMUNCULUS_HOME` (default `~/.homunculus`) holds `targets.json` (targets,
their tokens, and the session bookkeeping; mode 0600) and, while it runs, the
demo service's files under `demo/`. `homunculus demo stop` removes the demo's files
and targets; when no targets remain, `targets.json` and an empty home directory
are removed too. A `homunculus demo start` that fails, or is ended with Ctrl-C or
`SIGTERM` before it reports the service, leaves no service and no files.

`images/` holds the camera images `homunculus observe` saved without `--save`,
and the pictures `homunculus install` took. Nothing removes them, and a home
that holds them is not removed.

The robots that rigs download are not in `$HOMUNCULUS_HOME`. They are in the
`mujoco-menagerie` package's cache (see
[Where the robot comes from](#where-the-robot-comes-from)), and no homunculus
command removes them.

`homunculus demo stop` removes nothing while a process holds the demo's files
without answering as its service (a suspended or hung service, for example).
It exits with status 4 and names the process its record points to;
`homunculus demo status` reports the same.

Commands can overlap, for example `homunculus events --follow` in one shell while
you work in another: each command writes back to `targets.json` only what it
changed, and commands save one at a time, so two that finish in the same
instant both keep what they recorded. Two `homunculus demo start` at once give one
running demo; the other command exits with status 4. A `homunculus demo stop` that
overtakes a `homunculus demo start` wins: the start reports that the demo was
stopped while it was starting.

## Limitations

- A prototype for validating the interface. No physical robot, vendor SDK, or
  native controller is connected. The tests run on macOS arm64 with Python 3.11,
  3.13, 3.14 and 3.15.0rc2, and on Linux x86_64 (Debian, in a container) with
  Python 3.11 and 3.15.2. It needs a POSIX system, for file locks and signals; Windows is not
  supported.
- The fake robots prove protocol behavior only. `fake:g1` does not simulate
  balance or gait and says so in its description.
- Plain HTTP on loopback with static bearer tokens. Do not expose it to a
  network as is.
- Live video sends each frame as a whole JPEG image, 0.87 MB/s for one viewer
  of the Panda's `overview` camera. It was watched in Chromium only. Frames are
  rendered in the service's process, beside the control loop (10 ms period).
  The longest interval between control ticks, over a service's first 33 s with
  the viewers watching for 30 s of them, was 18.8 ms with no viewer, 36.7 ms
  with one, and 36.9 ms with four, with no execution lag. MuJoCo rendering is
  tested on macOS, and on Linux x86_64 with Mesa's software renderer.
- A rig's robot is downloaded from GitHub the first time, so a rig's first
  start needs the network.
- The MuJoCo backend does not compensate gravity, needs a keyframe named
  `home`, and refuses a camera set by sensor size. For those reasons 11 of
  Menagerie's 28 arms do not pass `homunculus install` as they are (see
  [Menagerie's arms](#menageries-arms)).
- Saved camera images are never removed from `$HOMUNCULUS_HOME/images`.
- Stream updates are separate HTTP requests; a persistent update channel
  is reported as unavailable. No MCP adapter yet.
- Profiles implemented: `joints.trajectory/1`, `gripper.width/1`,
  `hand.posture/1`, `base.planar_velocity/1`. Tool-pose and contact-qualified
  gripper completion are not, and are rejected as unsupported.
- The task adapter's tracking tolerance was qualified on recorded walks of one
  task. It catches a base that stops against an obstacle; a slow sideways push
  (under about 0.1 m/s) that slides along the obstacle can still pass.
- Target names come from robot IDs, so connecting a second service that serves
  the same robot IDs replaces the first one's targets.
- A service that accepts connections but does not answer (a suspended process,
  for example) makes a command wait 15 s, and `homunculus stop` 5 s, before it
  reports that the service did not answer. The command is not sent a second
  time. For a goal, a management operation, or a session open, the error names
  the request ID: repeating the command with `--request-id ID` returns the
  first attempt instead of running it twice. A command interrupted with Ctrl-C
  during that wait says the same. `homunculus events --follow` keeps waiting once
  it is connected.
- The journal keeps 10,000 requests per robot. Finished requests of earlier
  execution contexts make room; when the current context alone fills it, new
  requests are refused with `journal_full` until the service is restarted.
  Stopping is never refused for this reason.
- The record of a stop operation is kept in memory only: the newest 256
  finished operations per robot, and not across a restart. After that
  `homunculus operation inspect` answers `not_found` for it.
