"""Rigs: the robot bodies the MuJoCo backend serves.

A rig is a folder that holds `rig.toml` and the MJCF model it names. rig.toml says what MJCF does not: where the
robot's own files come from, which joints are the arm and the gripper, which cameras are the robot's own, and
which cameras are fixed to the robot's bodies. A rig is data. Nothing in it runs.

The folders beside this file are the rigs that ship with homunculus, each named by its folder:
`homunculus demo start so101`. A rig of your own is named by its path: `homunculus demo start ./my-arm`.

`menagerie` names a robot in MuJoCo Menagerie. The `mujoco-menagerie` package downloads its files, checks them,
and pins them to one revision; the backend loads them beside the rig's own files, so the rig's model can
`<include>` them. A rig without `menagerie` holds every file its model needs.

This module reads rigs and nothing more, so `homunculus rigs` works without MuJoCo. backends/mujoco.py loads
and checks them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field, ValidationError

from ..models import Model
from ..runtime import StartupError

SHIPPED = Path(__file__).parent
# What to do when MuJoCo is not installed, for every command that needs it.
NEEDS_MUJOCO = (
    "rigs and the mujoco backend need homunculus's `mujoco` extra, for example:\n"
    '  uv tool install "homunculus[mujoco] @ git+https://github.com/kyle-compute/homunculus"'
)


class Mount(Model):
    """A camera fixed to one of the robot's bodies, in that body's frame, as MuJoCo states a camera."""

    camera: str
    body: str
    pos: list[float] = Field(min_length=3, max_length=3)
    quat: list[float] = Field(min_length=4, max_length=4)  # w x y z; the camera looks along its -z, with y up
    fovy: float = Field(gt=0, lt=180)  # vertical field of view, in degrees


class Rig(Model):
    """What rig.toml holds."""

    description: str
    menagerie: str | None = None  # the MuJoCo Menagerie robot whose files load beside this folder's
    model: str = Field(pattern=r"^[^/]+$")  # an MJCF file in this folder, not in a folder below it
    arm: list[str] = Field(min_length=1)
    gripper: list[str] = []  # finger joints that slide; none means the rig has no gripper.width/1
    cameras: list[str] = []  # the model cameras that are the robot's own; the others are for spectators
    mount: list[Mount] = []


def names() -> list[str]:
    """The rigs that ship with homunculus."""
    return sorted(path.parent.name for path in SHIPPED.glob("*/rig.toml"))


def find(rig: str) -> Path:
    """The folder of the rig a command names.

    A path has a slash or starts with a dot, as `./my-arm` does. Anything else is the name of a shipped rig, in
    any case: `SO101` is `so101`.
    """
    if "/" in rig or rig.startswith("."):
        folder = Path(rig).expanduser().resolve()
        if not (folder / "rig.toml").is_file():
            raise StartupError(f"{folder} is not a rig: it has no rig.toml")
        return folder
    if rig.lower() not in names():
        raise StartupError(
            f"no rig named {rig!r}; homunculus ships {', '.join(names())}. "
            f"A rig folder of your own is named by its path, such as ./{rig}"
        )
    return SHIPPED / rig.lower()


def load(folder: Path) -> Rig:
    """Read a rig folder's rig.toml. A file the backend could not use is refused here, with the reason."""
    path = folder / "rig.toml"
    try:
        return Rig.model_validate(tomllib.loads(path.read_text()))
    except tomllib.TOMLDecodeError as error:
        raise StartupError(f"{path} is not TOML: {error}") from None
    except ValidationError as error:
        first = error.errors()[0]
        raise StartupError(f"{path}: {'.'.join(str(part) for part in first['loc'])}: {first['msg']}") from None
