#!/usr/bin/env python3

import os
import json
import numpy as np
import struct
import pickle
from typing import Any, List, Tuple, Optional

from motion_matching.core.features import build_motion_matching_features


# When a best match and its same-phase frame in the adjacent mirrored range
# are numerically indistinguishable, prefer the lower global frame index.  All
# runtime motion databases store mirrored clips as adjacent, equal-length
# ranges.  The epsilon is deliberately much smaller than meaningful query-cost
# differences observed in locomotion (roughly 1e-2 and above).
MIRROR_TIE_EPSILON = 1e-4


def _mirror_frame_index(
    frame: int,
    range_starts: np.ndarray,
    range_stops: np.ndarray,
) -> Optional[int]:
    """Return the same-offset frame in the adjacent mirrored range.

    Ranges are paired ``(0, 1), (2, 3), ...``.  A pair must exist and have
    equal lengths; otherwise no mirror relationship is inferred.
    """
    range_idx = int(np.searchsorted(range_starts, frame, side="right") - 1)
    if range_idx < 0 or range_idx >= len(range_starts):
        return None
    if not (int(range_starts[range_idx]) <= frame < int(range_stops[range_idx])):
        return None

    mirror_range_idx = range_idx + 1 if range_idx % 2 == 0 else range_idx - 1
    if mirror_range_idx < 0 or mirror_range_idx >= len(range_starts):
        return None

    range_length = int(range_stops[range_idx] - range_starts[range_idx])
    mirror_length = int(range_stops[mirror_range_idx] - range_starts[mirror_range_idx])
    if range_length != mirror_length:
        return None

    offset = frame - int(range_starts[range_idx])
    return int(range_starts[mirror_range_idx]) + offset


def _prefer_earlier_mirror_on_tie(
    best_index: int,
    costs: np.ndarray,
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    epsilon: Optional[float],
) -> Tuple[int, float]:
    """Apply deterministic mirror tie-breaking to an already-masked cost array."""
    best_index = int(best_index)
    if epsilon is None or epsilon < 0.0:
        return best_index, float(costs[best_index])

    mirror_index = _mirror_frame_index(best_index, range_starts, range_stops)
    if mirror_index is None or mirror_index >= len(costs):
        return best_index, float(costs[best_index])

    # Masked-out candidates have infinite cost and must never be revived by
    # the tie-break (for example, a mirror outside custom_search_ranges).
    mirror_cost = float(costs[mirror_index])
    best_cost = float(costs[best_index])
    if not np.isfinite(mirror_cost):
        return best_index, best_cost

    if abs(mirror_cost - best_cost) <= epsilon:
        chosen = min(best_index, mirror_index)
        return chosen, float(costs[chosen])
    return best_index, best_cost


class MotionMatchingDatabase:
    def __init__(self, search_start_ground_height: float = 0.0, search_end_ground_height: float = 0.0) -> None:
        self.bone_positions = None
        self.bone_velocities = None
        self.bone_rotations = None
        self.bone_angular_velocities = None
        self.bone_parents = None
        self.range_starts = None
        self.range_stops = None
        self.features = None
        self.features_offset = None
        self.features_scale = None
        self.contact_states = None
        self.custom_search_ranges = (
            None  # List of (start_searching_frame, end_searching_frame, end_of_motion_frame) tuples
        )

        self.terrain_meshes = []
        self.terrain_metadata = []  # List of dicts containing metadata from .json files

        self.search_start_ground_height = search_start_ground_height
        self.search_end_ground_height = search_end_ground_height

        # By default, the ground fit will start immediately
        self.start_ground_fit_distance = -1.0

    def load_from_file(self, filename: str) -> None:
        """Load database file"""
        print(f"Loading database from: {filename}")

        with open(filename, "rb") as f:
            # Read bone_positions
            rows = struct.unpack("I", f.read(4))[0]
            cols = struct.unpack("I", f.read(4))[0]
            data = np.frombuffer(f.read(rows * cols * 3 * 4), dtype=np.float32)
            self.bone_positions = data.reshape((rows, cols, 3))

            # Read bone_velocities
            rows = struct.unpack("I", f.read(4))[0]
            cols = struct.unpack("I", f.read(4))[0]
            data = np.frombuffer(f.read(rows * cols * 3 * 4), dtype=np.float32)
            self.bone_velocities = data.reshape((rows, cols, 3))

            # Read bone_rotations
            rows = struct.unpack("I", f.read(4))[0]
            cols = struct.unpack("I", f.read(4))[0]
            data = np.frombuffer(f.read(rows * cols * 4 * 4), dtype=np.float32)
            self.bone_rotations = data.reshape((rows, cols, 4))

            # Read bone_angular_velocities
            rows = struct.unpack("I", f.read(4))[0]
            cols = struct.unpack("I", f.read(4))[0]
            data = np.frombuffer(f.read(rows * cols * 3 * 4), dtype=np.float32)
            self.bone_angular_velocities = data.reshape((rows, cols, 3))

            # Read bone_parents
            size = struct.unpack("I", f.read(4))[0]
            self.bone_parents = np.frombuffer(f.read(size * 4), dtype=np.int32)

            # Read range_starts
            size = struct.unpack("I", f.read(4))[0]
            self.range_starts = np.frombuffer(f.read(size * 4), dtype=np.int32)

            # Read range_stops
            size = struct.unpack("I", f.read(4))[0]
            self.range_stops = np.frombuffer(f.read(size * 4), dtype=np.int32)

            # Read contact_states
            rows = struct.unpack("I", f.read(4))[0]
            cols = struct.unpack("I", f.read(4))[0]
            data = np.frombuffer(f.read(rows * cols * 1), dtype=np.uint8)
            self.contact_states = data.reshape((rows, cols)).astype(bool)

    def load_features_from_file(self, filename: str) -> None:
        """Load features file"""
        print(f"Loading features from: {filename}")

        with open(filename, "rb") as f:
            # Read features matrix
            nframes = struct.unpack("I", f.read(4))[0]
            nfeatures = struct.unpack("I", f.read(4))[0]
            features = np.frombuffer(f.read(nframes * nfeatures * 4), dtype=np.float32)
            self.features = features.reshape((nframes, nfeatures))

            # Read offset and scale
            struct.unpack("I", f.read(4))[0]
            self.features_offset = np.frombuffer(f.read(nfeatures * 4), dtype=np.float32)
            struct.unpack("I", f.read(4))[0]
            self.features_scale = np.frombuffer(f.read(nfeatures * 4), dtype=np.float32)

        # Fix very small scale values to avoid division overflow
        self.features_scale = self.features_scale.copy()
        self.features_scale[self.features_scale < 1e-8] = 1.0

    def load_terrains(self, filenames: List[str]) -> None:
        """
        Load terrain from an .obj file using the trimesh library.

        Args:
            filenames (List[str]): List of paths to the .obj files.

        Returns:
            trimesh.Trimesh: The loaded mesh object.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh library is required to load terrain. Please install it via pip.")

        for filename in filenames:
            if os.path.exists(filename):
                # Load the mesh (it will parse everything including vn and vt)
                mesh_loaded = trimesh.load(filename, force="mesh")
                # Create a new mesh with only vertices and faces, ignoring normals and texture coordinates
                # This will automatically merge duplicate vertices that share the same position
                mesh = trimesh.Trimesh(vertices=mesh_loaded.vertices, faces=mesh_loaded.faces)
                # Merge vertices that share the same position to get the original 8 vertices
                mesh.merge_vertices()
                self.terrain_meshes.append(mesh)

                # Load corresponding .json file if it exists
                json_filename = os.path.splitext(filename)[0] + ".json"
                if os.path.exists(json_filename):
                    with open(json_filename, "r") as f:
                        metadata = json.load(f)
                    self.terrain_metadata.append(metadata)
                else:
                    raise FileNotFoundError(f"JSON file {json_filename} not found for terrain file {filename}")
            else:
                raise FileNotFoundError(f"Terrain file {filename} not found.")

        assert len(self.terrain_meshes) == len(self.range_starts)

    def find_terrain_mesh(self, frame_idx: int) -> Optional[Tuple[Any, Any]]:
        for i, range_start in enumerate(self.range_starts):
            if frame_idx >= range_start and frame_idx < self.range_stops[i]:
                return self.terrain_meshes[i], self.terrain_metadata[i]
        return None

    def nframes(self) -> int:
        return self.bone_positions.shape[0] if self.bone_positions is not None else 0

    def nfeatures(self) -> int:
        return self.features.shape[1] if self.features is not None else 0

    def nbones(self) -> int:
        return self.bone_positions.shape[1] if self.bone_positions is not None else 0

    def nranges(self) -> int:
        return len(self.range_starts) if self.range_starts is not None else 0

    def build_motion_matching_features(
        self,
        bone_left_foot: int,
        bone_right_foot: int,
        bone_hips: int,
        traj_axis1: int = 0,
        traj_axis2: int = 2,
        forward: np.ndarray = np.array([0, 0, 1]),
    ) -> None:
        all_features, features_offset, features_scale, _ = build_motion_matching_features(
            bone_positions=self.bone_positions,
            bone_velocities=self.bone_velocities,
            bone_rotations=self.bone_rotations,
            bone_angular_velocities=self.bone_angular_velocities,
            bone_parents=self.bone_parents,
            range_starts=self.range_starts,
            range_stops=self.range_stops,
            bone_left_foot=bone_left_foot,
            bone_right_foot=bone_right_foot,
            bone_hips=bone_hips,
            traj_axis1=traj_axis1,
            traj_axis2=traj_axis2,
            forward=forward,
        )
        self.features = all_features
        self.features_offset = features_offset
        self.features_scale = features_scale
        self.features_scale[self.features_scale < 1e-8] = 1.0

    def save_features_to_pkl_file(self, filename: str) -> None:
        features = {
            "features": self.features,
            "features_offset": self.features_offset,
            "features_scale": self.features_scale,
        }

        with open(filename, "wb") as f:
            pickle.dump(features, f)

    def load_features_from_pkl_file(self, filename: str) -> None:
        with open(filename, "rb") as f:
            features = pickle.load(f)
        self.features = features["features"]
        self.features_offset = features["features_offset"]
        self.features_scale = features["features_scale"]

    def set_custom_search_ranges(self, ranges: Optional[List[Tuple[int, int, int]]]) -> None:
        """
        Set custom search ranges for motion matching.

        Args:
            ranges: List of tuples like [(start_searching_frame, end_searching_frame, end_of_motion_frame), ...]

        """
        if ranges is None:
            print("Clearing custom search ranges")
            self.custom_search_ranges = None
            return

        self.custom_search_ranges = ranges


def motion_matching_search_w_custom_ranges(
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    features: np.ndarray,
    features_offset: np.ndarray,
    features_scale: np.ndarray,
    query_normalized: np.ndarray,
    ignore_range_end: int = 20,
    custom_search_ranges: Optional[List[Tuple[int, int, int]]] = None,
    mirror_tie_epsilon: Optional[float] = MIRROR_TIE_EPSILON,
) -> Tuple[int, float, Optional[int], Optional[int]]:
    """
    Args:
        custom_search_ranges: List of tuples like [(start_searching_frame, end_searching_frame, end_of_motion_frame), ...]
    """
    nranges = len(range_starts)
    nframes = features.shape[0]

    # Create mask to mark frames to search
    search_mask = np.ones(nframes, dtype=bool)

    # 0. If custom search ranges are set, first limit the search range
    if custom_search_ranges is not None and len(custom_search_ranges) > 0:
        # Only search within custom ranges
        search_mask.fill(False)
        for start, end, end_of_motion in custom_search_ranges:
            # Ensure not out of array boundaries
            start = max(0, min(start, nframes))
            end = max(0, min(end, nframes))
            if start < end:
                search_mask[start:end] = True

    # 1. Handle ignore_range_end - exclude frames at the end of the range
    for r in range(nranges):
        range_stop = range_stops[r]
        # Exclude frames at the end of the range due to ignore_range_end
        if range_stop > ignore_range_end:
            range_end = range_stop - ignore_range_end
            # Exclude frames at the end of the range
            search_mask[range_end:range_stop] = False

    # 3. Calculate cost for all frames (vectorized)
    # Calculate feature distance
    feature_costs = ((features - query_normalized) ** 2).sum(axis=1)

    costs = feature_costs.copy()

    # 5. Apply search mask
    costs[~search_mask] = np.inf

    # 6. Find the index of the minimum cost
    best_index = np.argmin(costs)
    best_index, best_cost = _prefer_earlier_mirror_on_tie(
        best_index=best_index,
        costs=costs,
        range_starts=range_starts,
        range_stops=range_stops,
        epsilon=mirror_tie_epsilon,
    )

    # 7. Find the index of the best_index in the custom_search_ranges range
    matched_range_idx = None
    matched_end_of_motion = None

    if custom_search_ranges is not None and len(custom_search_ranges) > 0:
        for idx, (start, end, end_of_motion) in enumerate(custom_search_ranges):
            if start <= best_index < end:
                matched_range_idx = idx
                matched_end_of_motion = end_of_motion
                break

    return best_index, best_cost, matched_range_idx, matched_end_of_motion


def motion_matching_search_fixed(
    best_index: int,
    best_cost: float,
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    features: np.ndarray,
    features_offset: np.ndarray,
    features_scale: np.ndarray,
    query_normalized: np.ndarray,
    transition_cost: float = 0.0,
    ignore_range_end: int = 20,
    ignore_surrounding: int = 20,
    feature_mask: Optional[np.ndarray] = None,
    no_transition_cost_range: Optional[Tuple[int, int]] = None,
    mirror_tie_epsilon: Optional[float] = MIRROR_TIE_EPSILON,
) -> Tuple[int, float]:
    nranges = len(range_starts)
    curr_index = best_index
    nframes = features.shape[0]

    # Create all frame indices
    all_indices = np.arange(nframes)

    # Create mask to mark frames to search
    search_mask = np.ones(nframes, dtype=bool)

    # 1. Handle ignore_range_end - exclude frames at the end of the range
    for r in range(nranges):
        range_stop = range_stops[r]
        # Exclude frames at the end of the range due to ignore_range_end
        if range_stop > ignore_range_end:
            range_end = range_stop - ignore_range_end
            # Exclude frames at the end of the range
            search_mask[range_end:range_stop] = False

    # 2. Handle ignore_surrounding - skip frames around the current frame
    if curr_index != -1 and ignore_surrounding > 0:
        # Calculate the distance to the current frame
        distances = np.abs(all_indices - curr_index)
        # Exclude surrounding frames
        search_mask[distances < ignore_surrounding] = False
        search_mask[curr_index] = True

    # 3. Calculate cost for all frames (vectorized)
    diff = features - query_normalized
    if feature_mask is not None:
        diff = diff * feature_mask
    feature_costs = (diff**2).sum(axis=1)

    # 4. Add transition_cost to all frames except the current frame
    #    and except frames inside no_transition_cost_range (if provided).
    costs = feature_costs.copy()
    if curr_index != -1:
        transition_mask = all_indices != curr_index
        if no_transition_cost_range is not None:
            nt_start, nt_stop = no_transition_cost_range
            transition_mask &= ~((all_indices >= nt_start) & (all_indices < nt_stop))
        costs[transition_mask] += transition_cost

    # 5. Apply search mask
    costs[~search_mask] = np.inf

    # 6. Find the index of the minimum cost
    best_index = np.argmin(costs)
    best_index, best_cost = _prefer_earlier_mirror_on_tie(
        best_index=best_index,
        costs=costs,
        range_starts=range_starts,
        range_stops=range_stops,
        epsilon=mirror_tie_epsilon,
    )

    return best_index, best_cost


def database_search_fixed(
    best_index: int,
    best_cost: float,
    db: MotionMatchingDatabase,
    query: np.ndarray,
    transition_cost: float = 0.0,
    ignore_range_end: int = 20,
    ignore_surrounding: int = 20,
    custom_search_ranges: Optional[List[Tuple[int, int, int]]] = None,
    feature_mask: Optional[np.ndarray] = None,
    no_transition_cost_range: Optional[Tuple[int, int]] = None,
    mirror_tie_epsilon: Optional[float] = MIRROR_TIE_EPSILON,
) -> Any:
    # Normalize query
    query_normalized = (query - db.features_offset) / db.features_scale

    if custom_search_ranges is None:
        return motion_matching_search_fixed(
            best_index=best_index,  # Initial best_index
            best_cost=best_cost,  # Initial best_cost
            range_starts=db.range_starts,
            range_stops=db.range_stops,
            features=db.features,
            features_offset=db.features_offset,
            features_scale=db.features_scale,
            query_normalized=query_normalized,
            transition_cost=transition_cost,
            ignore_range_end=ignore_range_end,
            ignore_surrounding=ignore_surrounding,
            feature_mask=feature_mask,
            no_transition_cost_range=no_transition_cost_range,
            mirror_tie_epsilon=mirror_tie_epsilon,
        )
    else:
        return motion_matching_search_w_custom_ranges(
            range_starts=db.range_starts,
            range_stops=db.range_stops,
            features=db.features,
            features_offset=db.features_offset,
            features_scale=db.features_scale,
            query_normalized=query_normalized,
            ignore_range_end=ignore_range_end,
            custom_search_ranges=custom_search_ranges,
            mirror_tie_epsilon=mirror_tie_epsilon,
        )


if __name__ == "__main__":
    pass
