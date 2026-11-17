"""Which MuJoCo Menagerie arms does the MuJoCo backend drive as they are?

For every arm in Menagerie, this writes a rig by rule and runs the check on it, the one `homunculus install`
runs. The rule: Menagerie's own scene, and as arm joints every hinge or slide joint with a range that one
actuator drives directly. No cameras and no files of the rig's own. So a pass says the backend can drive that
arm's joints. It does not say that a finished rig exists for it: that takes a scene, cameras, and a look at
what the cameras see.

    uv run python examples/menagerie_sweep.py        # downloads 63 MB of models the first time

It prints one table row per arm, in Markdown.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
import mujoco_menagerie

from homunculus.backends.mujoco import check
from homunculus.runtime import StartupError

DRIVABLE = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        for robot in mujoco_menagerie.robots():
            if robot.category not in ("arm", "dual_arm"):
                continue
            model = mujoco.MjModel.from_xml_path(str(robot.xml()))  # the default scene
            drivers: dict[int, int] = {}  # joint -> how many actuators act on it directly
            for actuator in range(model.nu):
                if model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT:
                    joint = int(model.actuator_trnid[actuator, 0])
                    drivers[joint] = drivers.get(joint, 0) + 1
            arm = [
                model.joint(joint).name
                for joint in range(model.njnt)
                if int(model.jnt_type[joint]) in DRIVABLE and model.jnt_limited[joint] and drivers.get(joint) == 1
            ]
            folder = Path(temporary) / robot.name
            folder.mkdir()
            (folder / "rig.toml").write_text(
                f'description = "{robot.display_name}, as Menagerie ships it"\n'
                f'menagerie = "{robot.name}"\nmodel = "{robot.entry().file}"\narm = {json.dumps(arm)}\n'
            )
            driven = f"{len(arm)} of {model.nu}"
            try:
                steps = check(folder, Path(temporary) / "pictures")
            except StartupError as error:
                rows.append((robot.name, driven, "does not load", str(error).splitlines()[0]))
                continue
            failed = [step for step in steps if not step.passed]
            if failed:
                rows.append((robot.name, driven, f"fails: {failed[0].name}", failed[0].detail))
            else:
                rows.append((robot.name, driven, "passes", ""))

    print("| Menagerie arm | Joints driven, of its actuators | Check | Why |\n| --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    passed = sum(row[2] == "passes" for row in rows)
    print(f"\n{passed} of {len(rows)} pass.")


if __name__ == "__main__":
    main()
