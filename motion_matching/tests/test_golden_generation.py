"""Golden end-to-end generation test — the byte-identical refactor gate.

Runs single-config scenarios in-process (no worker pool) and asserts the
generated motion ``.npz``, terrain ``.obj``, and terrain ``.npy`` hash to fixed
golden values. Two scenarios are covered:

- ``low_speed_roll_78``: smallest single-config scenario; its first step is
  ``climb_76_up`` so it also loads/exercises the climb_76 database.
- ``low_speed_climb_76`` (first config only): pins the climb_76 skill output
  directly. The full scenario fans out to 40 configs (would spawn a pool); we
  generate just the first config in-process to keep the gate fast.

Production generation uses both NumPy's legacy global RNG for scenario recipe
durations and an unseeded ``np.random.default_rng()`` for terrain variants. To
lock every output here, the fixture seeds the global RNG and monkeypatches
``default_rng`` to a fixed seed.

These tests require the real databases under ``PHP_MOTION_DATABASE_DIR``; they are
skipped if absent.
"""

from __future__ import annotations

import pytest

from motion_matching import run as mm_run
from motion_matching.scenarios import builtin
from motion_matching.utils import DATABASE_DIR

from tests._hashing import hash_npz, hash_npy, hash_obj

# Golden content hashes captured from the FAR-pi source checkout and current
# downloaded databases on 2026-09-08. Regenerate intentionally ONLY if a
# numeric or database change is deliberately accepted — never merely to make
# the test pass.
#
# Each scenario maps to (scenario_fn, config_index, golden_npz, golden_obj, golden_npy_seeded).
_SCENARIOS = {
    "roll_78": (
        builtin.low_speed_roll_78,
        0,
        "09eee5f1b873a81370a53adbebf803737e95b67ca93f252e84f51a3a4a426db3",
        "f6fde926bd54f8662bc177d22903c2f326147a8d7ee251df0014cfd9b0c84223",
        "3a7d16b2b0fc57d15f239e32599bf37f703430eae32219608cab381b40cb515a",
    ),
    "climb_76_first_config": (
        builtin.low_speed_climb_76,
        0,
        "21106f348a772b5c487fc55a68c4b8b973b99b27ef091d18e6d7b9822b8c6e08",
        "e7aa7720b0b607e0f42d8a9db2a1e6934944c44b0fc9a5ee7b4f8789a4b446ff",
        "fa89823c0cf43b246cf5253a651be68c514f82ea2bc2a74d5ea3e050ebfb9f42",
    ),
}

_REQUIRED_DBS = ["database_g1_locomotion.bin", "database_g1_climb_76.bin", "database_g1_roll_78.bin"]

_dbs_present = all((DATABASE_DIR / name).is_file() for name in _REQUIRED_DBS)
pytestmark = pytest.mark.skipif(
    not _dbs_present, reason=f"golden test needs databases under {DATABASE_DIR}: {_REQUIRED_DBS}"
)


@pytest.fixture(scope="module")
def _seed_rng():
    """Force both NumPy RNG APIs to fixed seeds for the whole module.

    Module-scoped, so it can't use the function-scoped ``monkeypatch`` fixture;
    patch/restore manually.
    """
    import motion_matching.utils.data_utils as data_utils

    original_default_rng = data_utils.np.random.default_rng
    original_legacy_state = data_utils.np.random.get_state()
    data_utils.np.random.seed(0)
    data_utils.np.random.default_rng = lambda *_a, **_k: original_default_rng(0)
    try:
        yield
    finally:
        data_utils.np.random.default_rng = original_default_rng
        data_utils.np.random.set_state(original_legacy_state)


@pytest.fixture(scope="module")
def _feature_cache(tmp_path_factory):
    # Tests can read an immutable release bundle without adding feature caches
    # to it. Feature extraction and every golden output remain unchanged.
    cache_dir = tmp_path_factory.mktemp("features")
    original_db_path = mm_run.db_path
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            mm_run, "db_path",
            lambda relative: str(cache_dir / relative) if relative.endswith(".pkl") else original_db_path(relative),
        )
        yield


@pytest.fixture(scope="module")
def generated(tmp_path_factory, _seed_rng, _feature_cache):
    """Generate the first config of each scenario once; cache file paths by key."""
    outputs = {}
    for key, (scenario_fn, cfg_idx, *_goldens) in _SCENARIOS.items():
        configs, speed_up_ratio = scenario_fn()
        cfg = configs[cfg_idx]
        out_dir = tmp_path_factory.mktemp(key)
        required_skills = mm_run.collect_required_skills([cfg])
        mm_run.init_worker(required_skills=required_skills)
        mm_run.process_scenario(cfg, str(out_dir), force=True, speed_up_ratio=speed_up_ratio)
        base = out_dir / cfg["name"]
        outputs[key] = {
            "npz": str(base.with_name(base.name + "_motion.npz")),
            "obj": str(base.with_name(base.name + "_terrain.obj")),
            "npy": str(base.with_name(base.name + "_terrain.npy")),
        }
    return outputs


@pytest.mark.parametrize("key", list(_SCENARIOS))
def test_motion_npz_is_byte_identical(generated, key) -> None:
    assert hash_npz(generated[key]["npz"]) == _SCENARIOS[key][2]


@pytest.mark.parametrize("key", list(_SCENARIOS))
def test_terrain_obj_is_byte_identical(generated, key) -> None:
    assert hash_obj(generated[key]["obj"]) == _SCENARIOS[key][3]


@pytest.mark.parametrize("key", list(_SCENARIOS))
def test_terrain_npy_is_byte_identical_when_seeded(generated, key) -> None:
    # Locked only because the module-scoped fixture seeds default_rng(0).
    assert hash_npy(generated[key]["npy"]) == _SCENARIOS[key][4]
