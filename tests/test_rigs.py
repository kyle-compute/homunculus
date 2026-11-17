"""Rigs: the folders the MuJoCo backend serves, how a command names one, and what the check shows.

The tests that use a robot from MuJoCo Menagerie download it the first time they run, as the product does, and
read it from Menagerie's cache after that.
"""

import json
import shutil

import pytest

from homunculus import rigs
from homunculus.cli import main as cli_main
from homunculus.runtime import StartupError
from homunculus.server import load_backends

# Two joints of different kinds, each with its own position actuator, and a body for a camera to sit on. No
# gravity acts on either joint: the hinge turns about the vertical, and the slide moves level.
MODEL = """
<mujoco model="two joints">
  <compiler angle="radian"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.05" pos="0 0 -0.2" rgba="0.3 0.5 0.3 1"/>
    <body name="link">
      <joint name="swing" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0" mass="1" rgba="0.8 0.3 0.2 1"/>
      <body name="carriage" pos="0.3 0 0">
        <joint name="reach" type="slide" axis="1 0 0" range="0 0.2"/>
        <geom type="box" size="0.03 0.03 0.03" mass="0.5"/>
      </body>
    </body>
    <camera name="above" pos="0.2 0 1" fovy="50"/>
  </worldbody>
  <actuator>
    <position name="swing_motor" joint="swing" kp="50" kv="5"/>
    <position name="reach_motor" joint="reach" kp="400" kv="40"/>
  </actuator>
  <keyframe><key name="home" qpos="0 0"/></keyframe>
</mujoco>
"""
RIG = """
description = "two joints"
model = "two.xml"
arm = ["swing", "reach"]
"""
# The robot's own cameras: one the model has, and one the rig fixes to the carriage, looking straight down.
CAMERAS = """
cameras = ["above", "carried"]

[[mount]]
camera = "carried"
body = "carriage"
pos = [0, 0, 0.5]
quat = [1, 0, 0, 0]
fovy = 60
"""


def folder(tmp_path, rig: str = RIG, model: str = MODEL):
    """A rig folder of your own."""
    path = tmp_path / "my-arm"
    path.mkdir()
    (path / "rig.toml").write_text(rig)
    (path / "two.xml").write_text(model)
    return path


def test_every_shipped_rig_is_listed_by_its_folder_and_reads_as_a_rig():
    assert rigs.names() == ["panda", "so101", "ur5e"]
    for name in rigs.names():
        rig = rigs.load(rigs.find(name))
        assert (rigs.find(name) / rig.model).is_file()
        assert rig.menagerie  # none of them carries its robot's files


def test_a_command_names_a_shipped_rig_in_any_case_and_a_rig_of_your_own_by_its_path(tmp_path, monkeypatch):
    assert rigs.find("SO101") == rigs.find("so101") == rigs.SHIPPED / "so101"
    with pytest.raises(StartupError, match="no rig named 'so100'; homunculus ships panda, so101, ur5e"):
        rigs.find("so100")

    mine = folder(tmp_path)
    assert rigs.find(str(mine)) == mine
    monkeypatch.chdir(tmp_path)
    assert rigs.find("./my-arm") == mine
    # A bare word is always a shipped rig's name, never a folder that happens to be here.
    with pytest.raises(StartupError, match="no rig named 'my-arm'"):
        rigs.find("my-arm")
    with pytest.raises(StartupError, match=r"is not a rig: it has no rig\.toml"):
        rigs.find("./objects")


@pytest.mark.parametrize(
    "rig, says",
    [
        (RIG + 'scene = "desk"\n', "scene: Extra inputs are not permitted"),
        (RIG.replace('arm = ["swing", "reach"]', "arm = []"), "arm: List should have at least 1 item"),
        (RIG.replace('"two.xml"', '"models/two.xml"'), "model: String should match"),
        (RIG.replace('description = "two joints"', ""), "description: Field required"),
        (RIG + CAMERAS.replace("pos = [0, 0, 0.5]", "pos = [0, 0]"), "mount.0.pos: List should have at least 3"),
        (RIG + "arm = [", "is not TOML"),
    ],
)
def test_a_rig_file_the_backend_could_not_use_is_refused_with_the_reason(tmp_path, rig, says):
    with pytest.raises(StartupError, match=says):
        rigs.load(folder(tmp_path, rig))


@pytest.mark.parametrize(
    "rig, says",
    [
        (RIG.replace('"reach"', '"stretch"'), "has no joint named 'stretch'"),
        (RIG + 'cameras = ["wrist"]\n', "has no camera named 'wrist'; it has \\['above'\\]"),
        (RIG + CAMERAS.replace('"carriage"', '"hand"'), "mounts the camera 'carried' on the body 'hand'"),
        (RIG + 'menagerie = "franka_emika_pander"\n', "cannot get the robot 'franka_emika_pander' from MuJoCo"),
    ],
)
def test_a_rig_that_does_not_fit_its_model_is_refused_when_it_loads(tmp_path, rig, says):
    pytest.importorskip("mujoco")
    with pytest.raises(StartupError, match=says):
        load_backends("mujoco", {"rig": str(folder(tmp_path, rig))})


def test_a_rig_file_named_like_one_of_its_menagerie_robots_files_is_refused(tmp_path):
    """The two load side by side, so one would hide the other. A README or LICENSE of the rig's own is fine."""
    pytest.importorskip("mujoco")
    mine = folder(tmp_path, RIG + 'menagerie = "franka_emika_panda"\n')
    (mine / "README.md").write_text("my arm")
    assert list(load_backends("mujoco", {"rig": str(mine)})) == ["my-arm"]
    (mine / "panda.xml").write_text(MODEL)
    with pytest.raises(StartupError, match=r"panda\.xml is in the rig folder .* and in Menagerie's franka_emika_panda"):
        load_backends("mujoco", {"rig": str(mine)})


def test_a_rigs_revision_names_its_files_and_not_where_they_are(tmp_path):
    """Two machines that serve the same rig report the same revision. A change to any file changes it."""
    pytest.importorskip("mujoco")

    def revisions(rig) -> dict[str, str]:
        (backend,) = load_backends("mujoco", {"rig": str(rig)}).values()
        with backend:
            return backend.describe().revisions

    shipped = revisions("so101")
    assert shipped["menagerie"].startswith("robotstudio_so101@") and len(shipped["model_sha256"]) == 12
    assert revisions("so101") == shipped  # and not a new one each time it loads

    copy = shutil.copytree(rigs.find("so101"), tmp_path / "elsewhere")
    (copy / ".DS_Store").write_bytes(b"what macOS leaves in a folder")
    assert revisions(copy) == shipped

    with next(copy.glob("objects/*/model_collision_0.obj")).open("a") as mesh:
        mesh.write("# edited\n")
    edited = revisions(copy)
    assert edited["model_sha256"] != shipped["model_sha256"] and edited["menagerie"] == shipped["menagerie"]


def test_the_check_drives_a_rig_and_saves_what_each_of_its_cameras_sees(tmp_path):
    pytest.importorskip("mujoco")
    image_module = pytest.importorskip("PIL.Image")
    from homunculus.backends.mujoco import check

    steps = check(folder(tmp_path, RIG + CAMERAS), tmp_path / "pictures")
    assert [(step.name, step.passed) for step in steps] == [
        ("holds its start pose", True),
        ("each arm joint reaches a goal and returns", True),
        ("camera above takes a picture", True),
        ("camera carried takes a picture", True),
    ]
    # The mounted camera rides on the carriage, 0.5 m above it, and sees the floor it looks down on.
    carried = image_module.open(tmp_path / "pictures" / "my-arm-carried.jpg")
    assert carried.size == (640, 480)
    red, green, blue = carried.getpixel((40, 40))
    assert green > red and green > blue


# The hinge turned to swing in a vertical plane: gravity now loads it, and its actuator lets it sag.
SAGGING = MODEL.replace('name="swing" type="hinge" axis="0 0 1"', 'name="swing" type="hinge" axis="0 1 0"')
# A post in the carriage's way, 5 mm along its slide.
POST = '<geom name="post" type="box" size="0.03 0.03 0.1" pos="0.365 0 0"/>'
BLOCKED = MODEL.replace('<camera name="above"', f'{POST}\n    <camera name="above"')


@pytest.mark.parametrize(
    "model, fails, says",
    [
        (SAGGING, "holds its start pose", "swing moved 0.0"),
        (BLOCKED, "each arm joint reaches a goal and returns", "reach: out failed"),
    ],
)
def test_the_check_fails_a_rig_that_is_wired_correctly_but_does_not_work(tmp_path, model, fails, says):
    """Both models load. Only driving them shows the fault, and the check names the step and the joint."""
    pytest.importorskip("mujoco")
    from homunculus.backends.mujoco import check

    steps = check(folder(tmp_path, model=model), tmp_path / "pictures")
    failed = [step for step in steps if not step.passed]
    assert [step.name for step in failed] == [fails]
    assert says in failed[0].detail


@pytest.mark.parametrize("name", rigs.names())
def test_every_shipped_rig_passes_the_check(tmp_path, name):
    pytest.importorskip("mujoco")
    pytest.importorskip("PIL")
    from homunculus.backends.mujoco import check

    steps = check(rigs.find(name), tmp_path)
    assert [step for step in steps if not step.passed] == []
    assert len(list(tmp_path.glob(f"{name}-*.jpg"))) == len(rigs.load(rigs.find(name)).cameras)


def test_install_checks_a_rig_and_its_exit_status_says_whether_it_passed(tmp_path, monkeypatch, capsys):
    pytest.importorskip("mujoco")
    monkeypatch.setenv("HOMUNCULUS_HOME", str(tmp_path / "home"))
    mine = folder(tmp_path)

    assert cli_main(["install", str(mine)]) == 0
    text = capsys.readouterr().out
    assert "ok  holds its start pose" in text and f"ready: start it with `homunculus demo start {mine}`" in text

    (mine / "two.xml").write_text(BLOCKED)
    assert cli_main(["--json", "install", str(mine)]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["rig"] == "my-arm" and printed["passed"] is False
    assert [(step["name"], step["passed"]) for step in printed["steps"]] == [
        ("holds its start pose", True),
        ("each arm joint reaches a goal and returns", False),
    ]

    assert cli_main(["install", "so100"]) == 1
    assert "no rig named 'so100'" in capsys.readouterr().err


def test_rigs_lists_the_shipped_rigs_without_mujoco(capsys):
    assert cli_main(["--json", "rigs"]) == 0
    listed = json.loads(capsys.readouterr().out)["rigs"]
    assert [rig["name"] for rig in listed] == rigs.names()
    assert listed[1]["menagerie"] == "robotstudio_so101" and listed[1]["cameras"] == ["wrist_cam"]


@pytest.mark.parametrize(
    "arguments",
    [["so101", "--backend", "fake"], ["so101", "--backend-arg", "rig=panda"]],
)
def test_a_rig_given_with_another_backend_or_a_second_rig_is_a_usage_error(arguments, capsys):
    assert cli_main(["serve", *arguments]) == 2
    assert "names the rig for the mujoco backend to serve" in capsys.readouterr().err
