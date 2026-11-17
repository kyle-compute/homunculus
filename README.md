# homunculus

One command-line tool, `homunculus`, for driving different robots and simulators through the same
commands. You ask a robot what it can do, read what it measures, take control, request a
movement, and stop it. The commands are the same whichever backend runs the robot.

```console
$ homunculus goal joints joint4=-2.2 joint6=2.2 --duration 1.5 --wait
goal goal-cb9148d1d4b3: succeeded (completed)
  profile       joints.trajectory/1 on 'arm'; resources arm
  deadlines     30 s execution, 60 s wall
  measured      final_error={joint4=0, joint6=0}, peak_tracking_error=0, goal_tolerance=0.02
  evidence      observation obs-d59e34f2-1
  left holding  hold at joint1=0, joint2=0, joint3=0, joint4=-2.2, joint5=0, joint6=2.2,
                joint7=-0.7853
```

A goal succeeds only when the robot's own measurements show it finished. Elapsed time is not
enough. The result also says what the robot holds afterwards.

This is a prototype. **It has never been connected to a physical robot.**

## Backends

| Backend | Robots | What a result shows |
| --- | --- | --- |
| `fake` | `fake:arm` (the joints and gripper of the Franka Emika Panda) and `fake:g1` (walking base and hand), no physics | That the protocol behaves correctly |
| `mujoco` | `sim:arm`, the Franka Emika Panda arm and gripper from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) at a table with a scanned mug and bowl. Its cameras: `wrist` and `workspace`, plus `overview` for people watching | Simulated physics |
| `task` | One downloaded G1 benchmark task, run by its own simulator and graded by its own verifier | Simulated physics of that task |
| `MODULE:FUNCTION` | Your own robot or simulator: a `Backend` subclass, such as [`examples/lift.py`](examples/lift.py) | What your backend measures |

The MuJoCo backend also loads a model of your own: `--backend-arg model=arm.xml`, with the
joints to drive named in `arm=` and `gripper=`, and the robot's cameras in `cameras=`. [Your own robot](docs/guide.md#your-own-robot)
in the guide covers both.

## Install

Needs Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). No ROS, GPU, robot SDK or
credentials.

```sh
uv build --wheel
uv tool install dist/homunculus-0.1.0-py3-none-any.whl

# with the MuJoCo backend:
uv tool install "dist/homunculus-0.1.0-py3-none-any.whl[mujoco]"
```

`pip install dist/homunculus-0.1.0-py3-none-any.whl` works as well. The wheel is 6.4 MB and
installs 37 MB, nearly all of it the Panda's meshes and the two scanned objects.

## Quick start

```sh
homunculus demo start        # local service with two fake robots; fake:arm is the default target
homunculus describe          # what the robot supports, its limits, and what it holds now
homunculus session open      # take control; nothing moves yet
homunculus observe           # latest measurements
homunculus goal joints joint4=-2.2 joint6=2.2 --duration 1.5 --wait
homunculus goal gripper 0.03 --wait
homunculus stop              # stop everything this session commands
homunculus demo stop         # stop the service and remove its files
```

Use `-t fake:g1` to pick another robot for one command, or `homunculus use fake:g1` to change the
default. The G1 has a walking base, so try a continuous velocity stream on it:

```sh
homunculus use fake:g1 && homunculus session open
homunculus stream open --forward 0.3 --validity 1   # each target is valid for 1 s
homunculus stream update --forward 0.2 --yaw 0.4    # replaces the target without stopping
homunculus stream close                             # or let it expire: the base stops either way
```

Run those lines together. If you type them one at a time, the 1 s target expires and the base
stops before your update arrives.

## What the agent sees

`homunculus observe --camera NAME` records the robot's own cameras, saves each image, and prints its
path with the lens and where the camera was. An agent opens the file to see it.

```console
$ homunculus -t sim:arm observe --camera wrist --camera workspace
  ...
  image wrist rgb 640x480 image/jpeg: /Users/you/.homunculus/images/wrist-rgb-img-ccf5a695-1.jpg
    lens [433, 433] px focal, centre [320, 240] px; camera at [0.4555, 0.1189, 0.2945] m,
    orientation xyzw [0.6919, 0.6928, 0.1421, 0.1454] in world (optical axes: x right, y down, z
    forward)
  image workspace rgb 640x480 image/jpeg: /Users/you/.homunculus/images/workspace-rgb-img-ccf5a695-2.jpg
    ...
```

Every backend states its images the same way, whatever produced them: undistorted, with focal
lengths and principal point in pixels, and the camera's pose at capture with optical axes. With
those, a pixel and its depth give a point in metres. A camera that is only for people watching
(`overview` on `sim:arm`) plays with `homunculus video` but is never in an observation.

A robot refuses what it does not support before anything is sent:

```console
$ homunculus -t fake:g1 goal joints joint4=-2.2
error: fake-g1 does not offer joints.trajectory/1 as a goal; it offers: base.planar_velocity/1 on 'base' (goal/stream), hand.posture/1 on 'left_hand' (goal) [unsupported_profile]
```

## Commands

| Command | Does |
| --- | --- |
| `homunculus demo start / stop / status` | Run a local demo service in the background |
| `homunculus targets`, `use`, `connect`, `forget` | Manage the robots the CLI knows about |
| `homunculus serve` | Run a service in the foreground |
| `homunculus describe` | Show profiles, limits, units, frames and current state |
| `homunculus observe [--camera NAME] [--save DIR]` | Read measurements; with `--camera`, save the robot's camera images and print their paths |
| `homunculus video CAMERA` | Print a link that plays a camera live in a browser, the robot's or a spectator's |
| `homunculus session open / renew / close` | Take, extend or release control |
| `homunculus goal joints / gripper / hand / base` | Request one finite movement |
| `homunculus goal submit --file move.json` | Request any movement written as JSON |
| `homunculus goal inspect / cancel` | Follow or stop the last goal |
| `homunculus stream open / update / inspect / close` | Continuous velocity targets that expire unless renewed |
| `homunculus stop [--operator]` | Stop everything and wait until standstill is measured |
| `homunculus manage recover / world.reset` | Operator operations the robot advertises |
| `homunculus events [--follow]` | Numbered runtime events |

Every command has `--help`. Useful flags on goals: `--wait`, `--dry-run` (check without
moving), `--request-id ID` (a retry never runs twice), `--max-execution S`.

## Output and exit codes

Text by default. With `--json`, a command prints one JSON document (`events` prints one per line) and errors are one JSON
document on stderr: `{"error": {"code", "message", "hint", "details"}}`.

| Exit | Meaning |
| --- | --- |
| 0 | Done, or accepted and still running |
| 1 | The robot failed, was cancelled, or a wait timed out; or `homunculus serve` could not start |
| 2 | Bad command line or input file |
| 3 | The service refused the request; the message ends with a code such as `[stale_context]` |
| 4 | Nothing could be sent: no target, no session, or the service is unreachable |
| 130 | Interrupted with Ctrl-C. An interrupted `--wait` leaves the robot moving and says how to stop it |

## Python client

The CLI is a thin layer over this client, and both use the same typed models.

```python
from homunculus.client import RobotClient, TargetStore
from homunculus.models import GripperWidthGoal

store = TargetStore()  # the targets `homunculus demo start` or `homunculus connect` saved
with RobotClient(store.get("fake:arm"), on_change=store.save) as robot:
    robot.open_session()
    robot.submit_goal(GripperWidthGoal(profile="gripper.width/1", target="gripper", width_m=0.03))
    goal = robot.wait_goal()
    print(goal.state, goal.result.measurements, goal.retained)
    robot.close_session()
```

A robot has one control session. If the CLI already holds it, `open_session()` is refused with
`session_conflict`. Leave that call out to work in the CLI's session.

## Limitations

- No physical robot, vendor SDK or native controller is connected.
- It needs a POSIX system. Windows is not supported.
- The fake robots show protocol behaviour only. `fake:g1` has no balance or gait.
- The service is plain HTTP on 127.0.0.1 with static tokens. Do not expose it to a network.
- A live video link lets anyone who has it watch, without a token, until it expires. Live video
  is a stream of whole JPEG images, 0.87 MB/s for one viewer of a Panda camera, and was watched
  in Chromium only.
- Implemented profiles: `joints.trajectory/1`, `gripper.width/1`, `hand.posture/1`,
  `base.planar_velocity/1`. Tool-pose movement and contact-based gripper completion are refused
  as unsupported.
- The `task` backend is tested only with `g1-dining-salt-mill`. It walks and grips but has no
  arm control, so it cannot solve that task.

## Documentation

- [Usage guide](docs/guide.md): streams, stopping and recovery, units and clocks, JSON request
  files, the MuJoCo and task backends, adding your own robot, running your own service, and
  where files are stored.

## Third-party files

`src/homunculus/assets/franka_emika_panda/` holds the Franka Emika Panda model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/franka_emika_panda),
commit `370c439`, copied unchanged under its Apache-2.0 license (`LICENSE` in that folder). Only
`homunculus_scene.xml` there is this project's: the table, the objects, and the cameras.

`src/homunculus/assets/google_scanned_objects/` holds a mug and a bowl from
[Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)
(meshes and textures, CC BY 4.0, by Google), as converted for MuJoCo by
[kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) (MIT),
commit `6ff8d27`, copied unchanged. See the `README.md` there.

## Development

```sh
uv sync --extra mujoco
uv run pytest                                    # contract tests: fake robots, examples/lift.py, MuJoCo
uv run ruff check && uv run ruff format --check
```

Code is in `src/homunculus/`: `cli.py` (commands and output), `client.py` (HTTP client and
saved targets), `server.py` (HTTP service and backend loading), `runtime.py` (sessions, goals,
streams, stops, and the `Backend` base class), `profiles.py` (what each motion profile means,
and the judges that decide success), `models.py` (shared types) and `backends/`.
