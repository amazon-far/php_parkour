"""I/O helpers shared by the generation and interactive-recording paths.

These collect the motion-save and terrain-save steps that were previously
copy-pasted between ``run.py``'s ``process_scenario`` (headless generation) and
``run_interactive`` (Viser recording).
"""

from motion_matching.io.motion_io import save_motion_npz
from motion_matching.io.terrain_io import (
    build_combined_terrain_mesh,
    randomize_offpath_terrain_variants,
    randomize_terrain_variants,
    sample_offpath_box_obstacles,
)

__all__ = [
    "save_motion_npz",
    "build_combined_terrain_mesh",
    "randomize_offpath_terrain_variants",
    "randomize_terrain_variants",
    "sample_offpath_box_obstacles",
]
