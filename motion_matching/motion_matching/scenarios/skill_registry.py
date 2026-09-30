"""Declarative configuration for every motion-matching skill database.

The runtime ``init_databases`` function consumes ``SKILL_REGISTRY`` and produces
a fully-loaded :class:`MotionMatchingDatabase` per skill.

For each entry, only ``search_ranges`` is mandatory. The other fields default
to convention-driven values derived from the skill name:

- ``bin`` defaults to ``f"database_g1_{family}.bin"`` where ``family`` is the
  ``family`` field (or, if unset, the skill name itself).
- ``features`` defaults to ``f"features_g1_{family}.pkl"`` (same family).
- ``terrains`` defaults to ``[]`` (no terrain). Most skills DO need terrain;
  declare it here.
- ``height_range`` defaults to ``(0.0, 0.0)`` (no ground-height constraint).
- ``start_ground_fit_distance`` defaults to ``None`` (use db's built-in default).

When a skill diverges from the convention (e.g. its bin file has a speed factor
in the name like ``database_g1_dash_vault_76_0.75x.bin``), set ``bin`` and/or
``features`` explicitly. When a skill shares its bin with siblings (e.g.
``step_up`` reads ``database_g1_step.bin``), set ``family`` so the convention
points at the shared file.

Helper top-of-box heights for ground-height constraints:
"""

# Top-of-box ground heights (used for ``height_range`` start/end).
CLIMB_58_TOP = 0.585843
CLIMB_76_TOP = 0.76382022
CLIMB_94_TOP = 0.93808989
CLIMB_134_TOP = 1.34
ROLL_78_TOP = 0.77902247
ROLL_90_TOP = 0.90471910
STEP_TOP = 0.35234731
DROP_92_TOP = 0.92


# Per-skill terrain lists. For skills that load the same pair multiple times
# (climb_58 loads its terrain pair 3x for variant indexing), declare the full
# list. Order matters — the database maintains terrain meshes by index.
_STEP_TERRAINS = ["terrain/step/box_mirror.obj", "terrain/step/box.obj"]
_CLIMB_58_TERRAINS = [
    "terrain/climb_58/box.obj",
    "terrain/climb_58/box_mirror.obj",
    "terrain/climb_58/box.obj",
    "terrain/climb_58/box_mirror.obj",
    "terrain/climb_58/box.obj",
    "terrain/climb_58/box_mirror.obj",
]
_CLIMB_76_TERRAINS = ["terrain/climb_76/box.obj", "terrain/climb_76/box_mirror.obj"]
_CLIMB_76_HIGH_SPEED_TERRAINS = ["terrain/climb_76_high_speed/box.obj", "terrain/climb_76_high_speed/box_mirror.obj"]
_CLIMB_94_TERRAINS = ["terrain/climb_94/box.obj", "terrain/climb_94/box_mirror.obj"]
_CLIMB_94_XY_TERRAINS = ["terrain/climb_94/box_xy_swapped.obj", "terrain/climb_94/box_xy_swapped_mirror.obj"]
_CLIMB_134_TERRAINS = ["terrain/climb_134/box.obj", "terrain/climb_134/box_mirror.obj"]
_ROLL_78_TERRAINS = ["terrain/roll_78/box.obj", "terrain/roll_78/box_mirror.obj"]
_ROLL_90_TERRAINS = ["terrain/roll_90/box.obj", "terrain/roll_90/box_mirror.obj"]
_ROLL_136_TERRAINS = ["terrain/roll_136/box.obj", "terrain/roll_136/box_mirror.obj"]
_HURDLE_TERRAINS = ["terrain/hurdle/box.obj", "terrain/hurdle/box_mirror.obj"]
_DASH_VAULT_TERRAINS = ["terrain/dash_vault/box.obj", "terrain/dash_vault/box_mirror.obj"]
_SPEED_VAULT_TERRAINS = ["terrain/speed_vault/box.obj", "terrain/speed_vault/box_mirror.obj"]
_CAT_VAULT_TERRAINS = ["terrain/cat_vault/box.obj", "terrain/cat_vault/box_mirror.obj"]
_JUMP_TERRAINS = ["terrain/jump/box.obj", "terrain/jump/box_mirror.obj"]
_DROP_92_TERRAINS = ["terrain/drop_92/box.obj", "terrain/drop_92/box_mirror.obj"]


# Sentinel used by ``init_databases`` to identify the bare locomotion database
# (returned separately from ``traversal_dbs``).
LOCOMOTION_KEY = "locomotion"


SKILL_REGISTRY = {
    # Bare locomotion DB — returned as `db` from init_databases() (not part of
    # traversal_dbs). No terrain. Search range is broad on purpose.
    LOCOMOTION_KEY: {
        "bin": "database_g1_locomotion.bin",
        "features": "features_g1_locomotion.pkl",
        "search_ranges": [(300, 320, 6000)],
    },
    # ---- Step family (uses database_g1_step.bin) ----
    "step": {
        "family": "step",
        "terrains": _STEP_TERRAINS,
        "search_ranges": [(700, 710, 830), (700 + 2100, 710 + 2100, 830 + 2100)],
    },
    "step_up": {
        "family": "step",
        "terrains": _STEP_TERRAINS,
        "height_range": (0.0, STEP_TOP),
        "search_ranges": [(700, 710, 758), (700 + 2100, 710 + 2100, 758 + 2100)],
    },
    "step_down": {
        "family": "step",
        "terrains": _STEP_TERRAINS,
        "height_range": (STEP_TOP, 0.0),
        "search_ranges": [(740, 770, 830), (740 + 2100, 770 + 2100, 830 + 2100)],
    },
    # ---- Step high-speed family ----
    "step_high_speed": {
        "family": "step_high_speed",
        "terrains": _STEP_TERRAINS,
        "search_ranges": [(583, 592, 678), (583 + 1750, 592 + 1750, 678 + 1750)],
    },
    "step_high_speed_up": {
        "family": "step_high_speed",
        "terrains": _STEP_TERRAINS,
        "height_range": (0.0, STEP_TOP),
        "search_ranges": [(583, 592, 605), (583 + 1750, 592 + 1750, 605 + 1750)],
        "start_ground_fit_distance": 0.6,
    },
    "step_high_speed_down": {
        "family": "step_high_speed",
        "terrains": _STEP_TERRAINS,
        "height_range": (STEP_TOP, 0.0),
        "search_ranges": [(616, 641, 678), (616 + 1750, 641 + 1750, 678 + 1750)],
    },
    # ---- Climb 58 family (search ranges differ per direction) ----
    "climb_58": {
        "family": "climb_58",
        "terrains": _CLIMB_58_TERRAINS,
        "search_ranges": [(1452, 1552, 1815), (1815, 1915, 2178)],
    },
    "climb_58_up": {
        "family": "climb_58",
        "terrains": _CLIMB_58_TERRAINS,
        "height_range": (0.0, CLIMB_58_TOP),
        "search_ranges": [(1452, 1552, 1690), (1815, 1915, 2053)],
    },
    "climb_58_down": {
        "family": "climb_58",
        "terrains": _CLIMB_58_TERRAINS,
        "height_range": (CLIMB_58_TOP, 0.0),
        "search_ranges": [(1702, 1732, 1815), (2065, 2095, 2178)],
    },
    # ---- Climb 76 family ----
    "climb_76": {
        "family": "climb_76",
        "terrains": _CLIMB_76_TERRAINS,
        "search_ranges": [(82, 137, 645), (82 + 674, 137 + 674, 645 + 674)],
    },
    "climb_76_up": {
        "family": "climb_76",
        "terrains": _CLIMB_76_TERRAINS,
        "height_range": (0.0, CLIMB_76_TOP),
        "search_ranges": [(70, 95, 411), (70 + 674, 95 + 674, 411 + 674)],
    },
    "climb_76_down": {
        "family": "climb_76",
        "terrains": _CLIMB_76_TERRAINS,
        "height_range": (CLIMB_76_TOP, 0.0),
        "search_ranges": [(515, 521, 645), (515 + 1750, 521 + 1750, 645 + 1750)],
    },
    # ---- Climb 76 high-speed (separate bins per direction; up uses high_speed terrain dir, down uses regular) ----
    "climb_76_high_speed_up": {
        "bin": "database_g1_climb_76_high_speed_up.bin",
        "features": "features_g1_climb_76_high_speed_up.pkl",
        "terrains": _CLIMB_76_HIGH_SPEED_TERRAINS,
        "height_range": (0.0, CLIMB_76_TOP),
        "search_ranges": [(70, 96, 240), (70 + 316, 96 + 316, 240 + 316)],
    },
    "climb_76_high_speed_down": {
        "bin": "database_g1_climb_76_high_speed_down.bin",
        "features": "features_g1_climb_76_high_speed_down.pkl",
        "terrains": _CLIMB_76_TERRAINS,  # NOTE: down uses non-high-speed terrain dir
        "height_range": (CLIMB_76_TOP, 0.0),
        "search_ranges": [(434, 445, 537), (434 + 561, 445 + 561, 537 + 561)],
    },
    # ---- Climb 94 family (down direction uses xy_swapped terrains) ----
    "climb_94_up": {
        "family": "climb_94",
        "terrains": _CLIMB_94_TERRAINS,
        "height_range": (0.0, CLIMB_94_TOP),
        "search_ranges": [(75, 86, 360), (695, 706, 980)],
    },
    "climb_94_down": {
        "family": "climb_94",
        "terrains": _CLIMB_94_XY_TERRAINS,
        "height_range": (CLIMB_94_TOP, 0.0),
        "search_ranges": [(360, 430, 580), (980, 1050, 1220)],
    },
    "climb_94_high_speed_up": {
        "bin": "database_g1_climb_94_high_speed.bin",
        "features": "features_g1_climb_94_high_speed.pkl",
        "terrains": _CLIMB_94_TERRAINS,
        "height_range": (0.0, CLIMB_94_TOP),
        "search_ranges": [(75 // 2, 86 // 2, 360 // 2), (695 // 2, 706 // 2, 980 // 2)],
    },
    "climb_94_high_speed_down": {
        "bin": "database_g1_climb_94_high_speed.bin",
        "features": "features_g1_climb_94_high_speed.pkl",
        "terrains": _CLIMB_94_XY_TERRAINS,
        "height_range": (CLIMB_94_TOP, 0.0),
        "search_ranges": [(360 // 2, 430 // 2, 580 // 2), (980 // 2, 1050 // 2, 1220 // 2)],
    },
    # ---- Climb 134 ----
    "climb_134_up": {
        "family": "climb_134",
        "terrains": _CLIMB_134_TERRAINS,
        "height_range": (0.0, CLIMB_134_TOP),
        "search_ranges": [(54, 60, 220), (54 + 600, 60 + 600, 220 + 600)],
    },
    # ---- Roll family ----
    "roll_78": {
        "family": "roll_78",
        "terrains": _ROLL_78_TERRAINS,
        "height_range": (ROLL_78_TOP, 0.0),
        "search_ranges": [(0, 35, 215), (216, 216 + 35, 431)],
    },
    "roll_90": {
        "family": "roll_90",
        "terrains": _ROLL_90_TERRAINS,
        "height_range": (ROLL_90_TOP, 0.0),
        "search_ranges": [(0, 30, 214), (215, 215 + 30, 427)],
    },
    "roll_136": {
        "bin": "database_g1_roll_136_high_speed.bin",
        "features": "features_g1_roll_136_high_speed.pkl",
        "terrains": _ROLL_136_TERRAINS,
        "height_range": (CLIMB_134_TOP, 0.0),
        "search_ranges": [(0, 6, 135), (136, 142, 271)],
    },
    # ---- Hurdle and vaults ----
    "hurdle": {
        "family": "hurdle",
        "terrains": _HURDLE_TERRAINS,
        "search_ranges": [(0, 40, 120), (155, 195, 155 + 120)],
    },
    "dash_vault": {
        "bin": "database_g1_dash_vault_76_0.75x.bin",
        "features": "features_g1_dash_vault_76_0.75x.pkl",
        "terrains": _DASH_VAULT_TERRAINS,
        "search_ranges": [(130, 145, 224), (307 + 130, 307 + 145, 307 + 224)],
    },
    "speed_vault": {
        "bin": "database_g1_speed_vault_0.7x.bin",
        "features": "features_g1_speed_vault_0.7x.pkl",
        "terrains": _SPEED_VAULT_TERRAINS,
        "search_ranges": [(5, 28, 111), (180, 203, 286)],
    },
    "cat_vault": {"family": "cat_vault", "terrains": _CAT_VAULT_TERRAINS, "search_ranges": [(74, 88, 149)]},
    # ---- Misc ----
    "jump_35": {"family": "jump_35", "terrains": _JUMP_TERRAINS, "search_ranges": [(0, 5, 33), (34, 34 + 5, 67)]},
    "drop_92": {
        "family": "drop_92",
        "terrains": _DROP_92_TERRAINS,
        "height_range": (DROP_92_TOP, 0.0),
        "search_ranges": [(0, 1, 108), (160, 161, 268)],
    },
}


def resolve_paths(skill_name: str, cfg: dict) -> tuple[str, str]:
    """Return (bin_filename, features_filename) for a skill, applying defaults.

    Convention: ``bin = f"database_g1_{family}.bin"``,
    ``features = f"features_g1_{family}.pkl"``.

    The ``family`` field defaults to the skill name itself if unset.
    """
    family = cfg.get("family", skill_name)
    bin_name = cfg.get("bin", f"database_g1_{family}.bin")
    features_name = cfg.get("features", f"features_g1_{family}.pkl")
    return bin_name, features_name
