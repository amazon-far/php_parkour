"""Installed G1 resources and writable database cache resolution."""

import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from motion_matching import utils as paths


def test_assets_are_inside_package() -> None:
    assert paths.ASSETS_DIR == paths.PKG_ROOT / "assets"
    assert paths.ROBOTS_DIR.is_dir()
    assert (paths.ROBOT_MOTIONS_DIR / "climb_76_high_speed_up.npz").is_file()


def test_both_g1_models_resolve_and_retain_exact_mesh_union() -> None:
    required = set()
    for model in (paths.G1_URDF, paths.G1_MJCF):
        assert Path(model).is_file()
        required.update(
            Path(mesh.get("file") or mesh.get("filename")).name
            for mesh in ET.parse(model).iter("mesh")
            if mesh.get("file") or mesh.get("filename")
        )
    retained = {p.name for p in (paths.ROBOTS_DIR / "unitree_description/meshes/g1").iterdir()}
    assert len(required) == 36
    assert retained == required


def test_g1_urdf_loads_visual_and_collision_meshes() -> None:
    from motion_matching.kinematics.urdf_kinematics import URDFKinematics

    kin = URDFKinematics(paths.G1_URDF, load_meshes=True, load_collision_meshes=True)
    assert kin.num_joints == 29
    assert kin.urdf.scene.geometry
    assert kin.urdf.collision_scene.geometry


def _database_path_in_subprocess(env, cwd):
    code = "from motion_matching.utils import DATABASE_DIR; import json; print(json.dumps(str(DATABASE_DIR)))"
    env = {**env, "PYTHONPATH": str(paths.PKG_ROOT.parent)}
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, check=True, capture_output=True, text=True)
    return Path(json.loads(result.stdout))


def test_default_database_cache_does_not_require_creation(tmp_path) -> None:
    env = os.environ.copy()
    env.pop("PHP_MOTION_DATABASE_DIR", None)
    assert _database_path_in_subprocess(env, tmp_path) == Path("~/.cache/php-parkour/databases").expanduser()


def test_database_cache_override_is_resolved_without_creating_it(tmp_path) -> None:
    env = {**os.environ, "PHP_MOTION_DATABASE_DIR": str(tmp_path / "custom-databases")}
    assert _database_path_in_subprocess(env, tmp_path) == tmp_path / "custom-databases"
    assert not (tmp_path / "custom-databases").exists()


def test_db_path_is_under_database_dir() -> None:
    assert Path(paths.db_path("foo.bin")) == paths.DATABASE_DIR / "foo.bin"
