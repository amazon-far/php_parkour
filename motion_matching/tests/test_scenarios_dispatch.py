"""Scenario dispatch tests.

``run_generation`` looks scenarios up with ``getattr(motion_matching.scenarios,
name)``. After replacing the star-import in ``scenarios/__init__.py`` with an
explicit list, this guards that every scenario function in ``builtin.py`` is
still resolvable that way (catches a name added to builtin.py but forgotten in
the explicit re-export).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import motion_matching.scenarios as scenarios
from motion_matching.scenarios import builtin

# Scenario functions are the public (non-underscore) module-level functions in
# builtin.py that are defined there (not imported helpers).
_SCENARIO_FNS = [
    name
    for name, obj in inspect.getmembers(builtin, inspect.isfunction)
    if not name.startswith("_") and obj.__module__ == builtin.__name__
]


def test_at_least_the_known_scenarios_exist() -> None:
    expected = {
        "random_locomotion", "low_speed_step", "high_speed_step",
        "low_speed_climb_58", "high_speed_climb_58", "low_speed_climb_76", "high_speed_climb_76",
        "low_speed_climb_94", "high_speed_climb_94", "low_speed_roll_78", "low_speed_roll_90",
        "high_speed_roll_134", "high_speed_hurdle", "high_speed_dash_vault", "high_speed_speed_vault",
        "high_speed_cat_vault", "high_speed_jump_35",
    }
    assert expected <= set(_SCENARIO_FNS)


def test_every_builtin_scenario_is_dispatchable() -> None:
    missing = [name for name in _SCENARIO_FNS if not hasattr(scenarios, name)]
    assert not missing, f"scenarios not re-exported for getattr dispatch: {missing}"


def test_dispatched_object_is_the_same_function() -> None:
    for name in _SCENARIO_FNS:
        assert getattr(scenarios, name) is getattr(builtin, name)


def test_all_lists_match_dispatchable_scenarios() -> None:
    # __all__ should cover every scenario fn (plus the 'builtin' re-export).
    exported_scenarios = set(scenarios.__all__) - {"builtin"}
    assert exported_scenarios == set(_SCENARIO_FNS)


def test_random_locomotion_splits_offpath_obstacles_evenly() -> None:
    configs, _ = builtin.random_locomotion()
    assert len(configs) == 100

    obstacle_configs = [config for config in configs if "offpath_obstacles" in config]
    obstacle_free_configs = [config for config in configs if "offpath_obstacles" not in config]
    assert len(obstacle_configs) == 50
    assert len(obstacle_free_configs) == 50

    for index, config in enumerate(configs):
        if index % 2 == 0:
            assert "offpath_obstacles" not in config
            continue

        terrain = config["offpath_obstacles"]
        assert terrain["seed"] == 42000 + index
        assert terrain["min_count"] == 5
        assert terrain["max_count"] == 15
        assert terrain["path_clearance"] == 0.35
        assert terrain["min_center_distance"] == 0.3
        assert terrain["max_path_distance"] == 1.5
        assert terrain["obstacle_gap"] == 0.05


def test_random_locomotion_is_stand_to_body_to_stand() -> None:
    configs, speed_up_ratio = builtin.random_locomotion()
    assert speed_up_ratio == builtin.WALKING_SPEEDUP
    assert len({config["name"] for config in configs}) == len(configs)

    allowed_commands = {
        builtin.VEL_CMD_MAPPING_REVERSE["forward"],
        builtin.VEL_CMD_MAPPING_REVERSE["left_45"],
        builtin.VEL_CMD_MAPPING_REVERSE["left_90"],
        builtin.VEL_CMD_MAPPING_REVERSE["right_45"],
        builtin.VEL_CMD_MAPPING_REVERSE["right_90"],
        builtin.VEL_CMD_MAPPING_REVERSE["run_forward"],
        builtin.VEL_CMD_MAPPING_REVERSE["run_left_45"],
        builtin.VEL_CMD_MAPPING_REVERSE["run_left_90"],
        builtin.VEL_CMD_MAPPING_REVERSE["run_right_45"],
        builtin.VEL_CMD_MAPPING_REVERSE["run_right_90"],
    }
    for index, config in enumerate(configs):
        assert config["name"].startswith(f"cfg{index:04d}_stand_2.0s_")
        assert config["name"].endswith("_stand_2.0s")
        assert 3 <= len(config["steps"]) <= 12  # two stands + at most ten skills

        opening, *body, closing = config["steps"]
        for stand in (opening, closing):
            assert stand["speed"] == 0.0
            assert stand["duration"] == 2.0
            assert stand["vel_cmd_idx"] == builtin.VEL_CMD_MAPPING_REVERSE["stand"]

        assert body
        assert all(step["vel_cmd_idx"] in allowed_commands for step in body)
        assert all(0.5 <= step["duration"] <= 3.0 for step in body)
        for step in body:
            command = builtin.VEL_CMD_MAPPING[step["vel_cmd_idx"]]
            expected_speed = builtin.RUNNING_SPEED if command.startswith("run_") else builtin.WALKING_SPEED
            assert step["speed"] == expected_speed
            if command.startswith("run_"):
                assert step["duration"] >= 1.2


def test_random_locomotion_expected_body_length_matches_recipe() -> None:
    configs, _ = builtin.random_locomotion(num_configs=10_000)
    body_lengths = np.array([len(config["steps"]) - 2 for config in configs])
    assert body_lengths.mean() == pytest.approx(5.35, abs=0.1)
