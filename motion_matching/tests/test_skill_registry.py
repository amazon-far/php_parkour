"""Tests for motion_matching.scenarios.skill_registry — declarative skill config."""

import pytest

from motion_matching.scenarios import skill_registry as sr


class TestResolvePaths:
    def test_locomotion_explicit(self) -> None:
        cfg = sr.SKILL_REGISTRY[sr.LOCOMOTION_KEY]
        bin_name, features_name = sr.resolve_paths(sr.LOCOMOTION_KEY, cfg)
        assert bin_name == "database_g1_locomotion.bin"
        assert features_name == "features_g1_locomotion.pkl"

    def test_family_default_uses_skill_name(self) -> None:
        # If a skill has no family/bin/features, defaults derive from skill name.
        bin_name, features_name = sr.resolve_paths("foo", {})
        assert bin_name == "database_g1_foo.bin"
        assert features_name == "features_g1_foo.pkl"

    def test_family_override_used_for_defaults(self) -> None:
        bin_name, features_name = sr.resolve_paths("step_up", {"family": "step"})
        assert bin_name == "database_g1_step.bin"
        assert features_name == "features_g1_step.pkl"

    def test_explicit_bin_wins_over_family(self) -> None:
        cfg = {"family": "step", "bin": "database_g1_custom.bin"}
        bin_name, features_name = sr.resolve_paths("step_up", cfg)
        assert bin_name == "database_g1_custom.bin"
        # features still derives from family.
        assert features_name == "features_g1_step.pkl"


class TestSkillRegistryInvariants:
    def test_locomotion_key_present(self) -> None:
        assert sr.LOCOMOTION_KEY in sr.SKILL_REGISTRY

    def test_every_skill_has_search_ranges(self) -> None:
        for name, cfg in sr.SKILL_REGISTRY.items():
            assert "search_ranges" in cfg, f"{name} missing search_ranges"
            assert len(cfg["search_ranges"]) > 0, f"{name} has empty search_ranges"

    @pytest.mark.parametrize("name", list(sr.SKILL_REGISTRY.keys()))
    def test_search_range_tuples_well_formed(self, name: str) -> None:
        cfg = sr.SKILL_REGISTRY[name]
        for rng in cfg["search_ranges"]:
            assert len(rng) == 3, f"{name}: ranges must be (start, mid, end), got {rng!r}"
            start, mid, end = rng
            assert start < mid <= end, f"{name}: bad ordering in range {rng!r}"

    def test_height_ranges_are_2_tuples(self) -> None:
        for name, cfg in sr.SKILL_REGISTRY.items():
            if "height_range" in cfg:
                hr = cfg["height_range"]
                assert len(hr) == 2, f"{name}: height_range must be (start, end)"

    def test_resolve_paths_succeeds_for_every_skill(self) -> None:
        # Every entry in the registry must be resolvable.
        for name, cfg in sr.SKILL_REGISTRY.items():
            bin_name, features_name = sr.resolve_paths(name, cfg)
            assert bin_name.endswith(".bin")
            assert features_name.endswith(".pkl")
