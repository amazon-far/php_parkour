#!/usr/bin/env python3
"""
Motion Matching integration and visualization using viser.
"""

import argparse
import multiprocessing
import os
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np

import motion_matching.utils.animation_state as mm_utils
import motion_matching.utils.math_utils as math
from motion_matching.core.database import MotionMatchingDatabase, database_search_fixed
from motion_matching.core.features import (
    forward_kinematics,
    forward_kinematics_velocity,
)
from motion_matching.io import (
    build_combined_terrain_mesh,
    randomize_offpath_terrain_variants,
    randomize_terrain_variants,
    sample_offpath_box_obstacles,
    save_motion_npz,
)
from motion_matching.metadata.constants import (
    G1_BONE_LEFT_FOOT as bone_left_foot,
)
from motion_matching.metadata.constants import (
    G1_BONE_RIGHT_FOOT as bone_right_foot,
)
from motion_matching.metadata.constants import (
    G1_JOINT_AXES as joint_axes,
)
from motion_matching.metadata.constants import (
    NUM_TERRAIN_VARIANTS as num_terrain_variants,
)
from motion_matching.scenarios.builtin import VEL_CMD_DIM
from motion_matching.scenarios.skill_registry import (
    LOCOMOTION_KEY,
    SKILL_REGISTRY,
    resolve_paths,
)
from motion_matching.utils import db_path
from motion_matching.utils.data_utils import (
    adjust_data_speed,
    adjust_vel_cmd_speed,
    resample_vel_cmd,
)

# --- Locomotion target speeds (interactive mode) ---
# Toggled by the speed-modifier key (Tab). Each value is the world-space target
# speed in m/s when the user holds a movement key.
LOCOMOTION_LOW_SPEED = 0.8
LOCOMOTION_HIGH_SPEED = 2.2

# Per-axis scale relative to the forward direction. Used to give the matching
# database different velocity targets when strafing or backpedalling so it
# selects from the appropriate clips.
SPEED_AXIS_SCALE_FORWARD = 1.0
SPEED_AXIS_SCALE_SIDE = 0.75
SPEED_AXIS_SCALE_BACK = 0.625

# --- Settled-stop snap ---
# Frames 0..627 of the locomotion DB are all one byte-identical static standing
# pose (verified: zero deviation in bone_rotations, bone_positions and
# bone_velocities across the whole range). It is the pose every stop should end
# on, so once the robot is genuinely standing there is nothing to search for --
# cut straight to it.
#
# Without this, deceleration ends in the matcher wandering among near-standing
# frames in other clips. Those poses differ from each other by up to 0.29 rad and
# from the standing pose by 0.49-0.57 rad, so the wandering is plainly visible,
# and each hop slides the feet.
STOP_FRAME = 0

# Only consider snapping once the command is idle. This gate is load-bearing:
# ordinary walking passes through double support, where both feet are briefly
# slow and level, and without the gate that would snap a walking robot to a
# standstill mid-stride.
STOP_IDLE_EPS = 0.01

# "Standing" means both feet are nearly still and the stance is roughly
# symmetric, measured on the live inertialized pose rather than the DB frame,
# because what matters is whether the robot itself has settled.
#
# Sizing, swept over 6 walk->release rollouts (low/high speed x 3 gait phases):
# the foot-speed test is the only binding one -- by the time both feet are below
# 0.3 m/s the stance is already inside even a 0.04/0.08 symmetry bound, so the
# symmetry terms cost nothing in latency and only serve to reject asymmetric
# poses that happen to be momentarily slow. Snapping fires 0.5-1.0s earlier than
# the search would have converged on its own; the worst gait phases, which
# previously wandered for 2.1-2.2s, gain the most.
STOP_FOOT_SPEED_EPS = 0.3
# Fore-aft split between the feet, root-local. The standing pose has 0.0.
STOP_STANCE_DX_EPS = 0.04
# Lateral asymmetry, root-local: |left_y + right_y|. The standing pose has 0.0
# with the feet at +-0.104.
STOP_STANCE_DY_EPS = 0.08


def _load_db(
    bin_name: str,
    feature_name: str,
    search_ranges: list,
    *,
    terrains: Optional[list] = None,
    height_range: tuple = (0.0, 0.0),
    start_ground_fit_distance: Optional[float] = None,
) -> MotionMatchingDatabase:
    """Load a MotionMatchingDatabase, build/load features, attach terrains.

    Args:
        bin_name: Filename of the binary motion-matching database (under
            ``PHP_MOTION_DATABASE_DIR``).
        feature_name: Filename of the feature pickle. Built (and cached to disk)
            on first call if absent; loaded on subsequent calls.
        search_ranges: List of (start_searching_frame, end_searching_frame,
            end_of_motion_frame) tuples passed to
            ``set_custom_search_ranges``.
        terrains: Optional list of OBJ filenames (under ``PHP_MOTION_DATABASE_DIR``) to
            load as terrain meshes. Default no terrain.
        height_range: ``(search_start_ground_height, search_end_ground_height)``
            passed to the DB constructor. Default ``(0.0, 0.0)``.
        start_ground_fit_distance: Optional override for the ground-fit start
            distance. Set to a positive float to delay ground fitting.
    """
    db = MotionMatchingDatabase(search_start_ground_height=height_range[0], search_end_ground_height=height_range[1])
    db.load_from_file(db_path(bin_name))
    if terrains:
        db.load_terrains([db_path(t) for t in terrains])

    feature_path = db_path(feature_name)
    if not os.path.exists(feature_path):
        db.build_motion_matching_features(
            bone_left_foot=bone_left_foot,
            bone_right_foot=bone_right_foot,
            bone_hips=1,
            traj_axis1=0,
            traj_axis2=1,
            forward=np.array([1, 0, 0]),
        )
        db.save_features_to_pkl_file(feature_path)
    else:
        db.load_features_from_pkl_file(feature_path)

    db.set_custom_search_ranges(search_ranges)
    if start_ground_fit_distance is not None:
        db.start_ground_fit_distance = start_ground_fit_distance
    return db


def init_databases(
    required_skills: Optional[Any] = None,
) -> tuple[MotionMatchingDatabase, dict[str, MotionMatchingDatabase], list[str]]:
    """Construct the locomotion DB and a configured set of traversal DBs.

    Args:
        required_skills: Optional iterable of skill names to load. If ``None``
            (default), every skill in :data:`SKILL_REGISTRY` is loaded — the
            full set needed for interactive mode. Pass a small set
            (e.g. ``{"step", "step_up", "step_down"}``) for generate mode where
            only one scenario's skills are needed; the locomotion DB is always
            included.

    Returns:
        (db, traversal_dbs, traversal_skills):
            - db: the bare locomotion :class:`MotionMatchingDatabase`.
            - traversal_dbs: ``{skill_name: db}`` for every loaded non-locomotion
              skill.
            - traversal_skills: list of loaded skill names.
    """
    if required_skills is not None:
        required_skills = set(required_skills)
        # Locomotion is always implicitly required.
        required_skills.add(LOCOMOTION_KEY)

    locomotion_db = None
    traversal_dbs = {}
    for skill_name, cfg in SKILL_REGISTRY.items():
        if required_skills is not None and skill_name not in required_skills:
            continue
        bin_name, feature_name = resolve_paths(skill_name, cfg)
        loaded = _load_db(
            bin_name=bin_name,
            feature_name=feature_name,
            search_ranges=cfg["search_ranges"],
            terrains=cfg.get("terrains", None),
            height_range=cfg.get("height_range", (0.0, 0.0)),
            start_ground_fit_distance=cfg.get("start_ground_fit_distance"),
        )
        if skill_name == LOCOMOTION_KEY:
            locomotion_db = loaded
        else:
            traversal_dbs[skill_name] = loaded

    traversal_skills = list(traversal_dbs.keys())
    return locomotion_db, traversal_dbs, traversal_skills


def collect_required_skills(scenarios: list[dict[str, Any]]) -> set[str]:
    """Walk a list of scenario configs and return the set of skill_names referenced.

    Used to load only the skills a given scenario will dispatch in generate
    mode, rather than the full 28-skill set.
    """
    skills = set()
    for cfg in scenarios:
        for step in cfg.get("steps", []):
            if step.get("type") == "skill":
                skills.add(step["skill_name"])
    return skills


class MotionMatchingSystem:
    def __init__(self, required_skills: Optional[Any] = None) -> None:
        self.db, self.traversal_dbs, self.traversal_skills = init_databases(required_skills)

        self.reset()

    def reset(self) -> None:
        # Initialize simulation state (full C++ style)
        # Get initial frame index (like C++)
        self.frame_idx = self.db.range_starts[0]  # Start from first range
        self.inertialize_blending_halflife = 0.1

        self.curr_bone_positions = self.db.bone_positions[self.frame_idx].copy()
        self.curr_bone_velocities = self.db.bone_velocities[self.frame_idx].copy()
        self.curr_bone_rotations = self.db.bone_rotations[self.frame_idx].copy()
        self.curr_bone_angular_velocities = self.db.bone_angular_velocities[self.frame_idx].copy()
        self.curr_bone_contacts = self.db.contact_states[self.frame_idx].copy()

        self.trns_bone_positions = self.db.bone_positions[self.frame_idx].copy()
        self.trns_bone_velocities = self.db.bone_velocities[self.frame_idx].copy()
        self.trns_bone_rotations = self.db.bone_rotations[self.frame_idx].copy()
        self.trns_bone_angular_velocities = self.db.bone_angular_velocities[self.frame_idx].copy()
        self.trns_bone_contacts = self.db.contact_states[self.frame_idx].copy()

        self.bone_positions = self.db.bone_positions[self.frame_idx].copy()
        self.bone_velocities = self.db.bone_velocities[self.frame_idx].copy()
        self.bone_rotations = self.db.bone_rotations[self.frame_idx].copy()
        self.bone_angular_velocities = self.db.bone_angular_velocities[self.frame_idx].copy()

        self.inertializer = mm_utils.InertializerState.create(self.db.nbones())

        # Initialize inertializer with current pose (like C++)
        mm_utils.inertialize_pose_reset(
            state=self.inertializer, root_position=self.bone_positions[0], root_rotation=self.bone_rotations[0]
        )

        mm_utils.inertialize_pose_update(
            bone_positions=self.bone_positions,
            bone_velocities=self.bone_velocities,
            bone_rotations=self.bone_rotations,
            bone_angular_velocities=self.bone_angular_velocities,
            state=self.inertializer,
            bone_input_positions=self.db.bone_positions[self.frame_idx],
            bone_input_velocities=self.db.bone_velocities[self.frame_idx],
            bone_input_rotations=self.db.bone_rotations[self.frame_idx],
            bone_input_angular_velocities=self.db.bone_angular_velocities[self.frame_idx],
            halflife=self.inertialize_blending_halflife,
            dt=0.0,
        )

        # Trajectory & Gameplay Data
        self.search_time = 0.1
        self.search_timer = self.search_time
        self.force_search_timer = self.search_time

        self.desired_velocity = np.zeros(3, dtype=np.float32)
        self.desired_velocity_change_curr = np.zeros(3, dtype=np.float32)
        self.desired_velocity_change_prev = np.zeros(3, dtype=np.float32)
        self.desired_velocity_change_threshold = 50.0

        self.desired_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.desired_rotation_change_curr = np.zeros(3, dtype=np.float32)
        self.desired_rotation_change_prev = np.zeros(3, dtype=np.float32)
        self.desired_rotation_change_threshold = 50.0

        self.simulation_position = np.zeros(3, dtype=np.float32)
        self.simulation_velocity = np.zeros(3, dtype=np.float32)
        self.simulation_acceleration = np.zeros(3, dtype=np.float32)
        self.simulation_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.simulation_angular_velocity = np.zeros(3, dtype=np.float32)

        self.simulation_velocity_halflife = 0.27
        self.simulation_rotation_halflife = 0.27

        # Scalar target speed (m/s) for locomotion in interactive mode. Set by
        # the main loop based on the speed-toggle state; kept on self so the
        # trajectory predictor can pick it up.
        self.target_speed = LOCOMOTION_LOW_SPEED

        self.trajectory_desired_velocities = np.zeros((4, 3), dtype=np.float32)
        self.trajectory_desired_rotations = np.zeros((4, 4), dtype=np.float32)
        self.trajectory_desired_rotations[:, 0] = 1.0  # Identity quaternions
        self.trajectory_positions = np.zeros((4, 3), dtype=np.float32)
        self.trajectory_velocities = np.zeros((4, 3), dtype=np.float32)
        self.trajectory_accelerations = np.zeros((4, 3), dtype=np.float32)
        self.trajectory_rotations = np.zeros((4, 4), dtype=np.float32)
        self.trajectory_rotations[:, 0] = 1.0  # Identity quaternions
        self.trajectory_angular_velocities = np.zeros((4, 3), dtype=np.float32)

        # Contact and Foot Locking data
        self.contact_bones = np.array([bone_left_foot, bone_right_foot], dtype=np.int32)
        self.contact_states = np.zeros((2), dtype=np.bool)
        self.contact_locks = np.zeros((2), dtype=np.bool)
        self.contact_positions = np.zeros((2, 3), dtype=np.float32)
        self.contact_velocities = np.zeros((2, 3), dtype=np.float32)
        self.contact_points = np.zeros((2, 3), dtype=np.float32)
        self.contact_targets = np.zeros((2, 3), dtype=np.float32)
        self.contact_offset_positions = np.zeros((2, 3), dtype=np.float32)
        self.contact_offset_velocities = np.zeros((2, 3), dtype=np.float32)

        # Reset contacts
        bone_pos, bone_vel, _, _ = forward_kinematics_velocity(
            bone_positions=self.bone_positions,
            bone_velocities=self.bone_velocities,
            bone_rotations=self.bone_rotations,
            bone_angular_velocities=self.bone_angular_velocities,
            bone_parents=self.db.bone_parents,
        )

        for i, contact_bone in enumerate(self.contact_bones):
            bone_position = bone_pos[contact_bone]
            bone_velocity = bone_vel[contact_bone]
            mm_utils.contact_reset(
                contact_state=self.contact_states[i : i + 1],
                contact_lock=self.contact_locks[i : i + 1],
                contact_position=self.contact_positions[i],
                contact_velocity=self.contact_velocities[i],
                contact_point=self.contact_points[i],
                contact_target=self.contact_targets[i],
                contact_offset_position=self.contact_offset_positions[i],
                contact_offset_velocity=self.contact_offset_velocities[i],
                input_contact_position=bone_position,
                input_contact_velocity=bone_velocity,
                input_contact_state=False,
            )

        self.motion_state = "locomotion"
        self.traversal_curr_db = None
        self.traversal_curr_frame = 0
        self.traversal_ending_frame = -1
        self.curr_ground_height = 0.0

        # Single-support root anchor: shifts bone_positions[0][:2] each tick so
        # the supporting foot's world xy tracks the DB's Sim-local foot delta
        # (rotated by the visualized yaw) instead of slipping with the
        # inertializer-blended root. Cleared on every state where correction
        # is invalid (double-support, flight, non-locomotion, toggle off).
        self._anchor_foot: Optional[int] = None
        self._anchor_world_xy = np.zeros(2, dtype=np.float32)
        self._anchor_prev_db_local_foot = np.zeros(3, dtype=np.float32)
        self._anchor_prev_frame_idx = -1
        self._anchor_correction_enabled = True
        self._anchor_max_dxy = 0.0
        self._anchor_last_log = 0.0
        # Persistent xy shift always applied to bone_positions[0]. Updated
        # during single-support; held (frozen) otherwise so the foot pin
        # doesn't snap off at heel-strike or stop. Zeroed after every
        # inertialize_pose_transition because the inertializer absorbs the
        # corrected position into transition_dst_position at that moment.
        self._root_xy_offset = np.zeros(2, dtype=np.float32)

        # Latched once the robot is judged to be standing, so the settle test is
        # only applied on the way in. Cleared by any non-idle command. See
        # _stop_settled() and STOP_FRAME.
        self.stopped = False

        self.terrain_meshes_to_add = []  # List of (mesh, pos, rot, metadata)

        self.ground_path_positions = []
        self.last_ground_path_position = None
        self.ground_path_segment_size = 0.5

    def _db_local_foot(self, frame: int, bone: int) -> np.ndarray:
        """Foot position at a DB frame, in the DB's Simulation-local frame.

        FK on ``db.bone_positions[frame]`` / ``db.bone_rotations[frame]`` walks
        from the Simulation anchor (index 0), so the returned position for
        ``bone`` is in Sim-local space — exactly what we need before rotating
        by the visualized character's yaw to get a world-frame delta.
        """
        global_positions = forward_kinematics(
            bone_positions=self.db.bone_positions[frame],
            bone_rotations=self.db.bone_rotations[frame],
            bone_parents=self.db.bone_parents,
        )
        return global_positions[bone].astype(np.float32)

    def _step_update_desired_velocity(
        self,
        *,
        dt: float,
        stick_direction: np.ndarray,
        camera_azimuth: float,
        desired_strafe: bool,
        fwrd_speed: float,
        side_speed: float,
        back_speed: float,
        is_auto_mode: bool,
        auto_desired_velocity: Optional[np.ndarray],
        auto_desired_rotation: Optional[np.ndarray],
        force_search_override: bool,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """Phase 1: update desired velocity/rotation and decide force_search.

        Returns ``(force_search, gamepadstick_left, gamepadstick_right)``. The
        two gamepad-stick vectors are only meaningful (and only consumed) on the
        interactive path; in auto mode they are returned as zero vectors.
        """
        if is_auto_mode and auto_desired_velocity is not None:
            self.desired_velocity = auto_desired_velocity.copy()
            self.desired_rotation = auto_desired_rotation.copy()

            # Calculate change for force search logic (optional for auto mode, but good to keep state consistent)
            self.desired_velocity_change_prev = self.desired_velocity_change_curr.copy()
            self.desired_velocity_change_curr = np.zeros(3)  # Assume smooth in auto
            self.desired_rotation_change_prev = self.desired_rotation_change_curr.copy()
            self.desired_rotation_change_curr = np.zeros(3)

            force_search = force_search_override
            gamepadstick_left = np.zeros(3, dtype=np.float32)
            gamepadstick_right = np.zeros(3, dtype=np.float32)
        else:
            # Interactive logic. stick_direction is a unit-or-zero direction
            # in world space; the matching pipeline treats it as a gamepad
            # stick value with magnitude == fraction of full speed.
            stick_norm = np.linalg.norm(stick_direction)
            if stick_norm > 1e-4:
                gamepadstick_left = (stick_direction / stick_norm).astype(np.float32)
            else:
                gamepadstick_left = np.zeros(3, dtype=np.float32)

            gamepadstick_right = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            desired_velocity_curr = mm_utils.desired_velocity_update(
                gamepadstick_left=gamepadstick_left,
                camera_azimuth=camera_azimuth,
                simulation_rotation=self.simulation_rotation,
                fwrd_speed=fwrd_speed,
                side_speed=side_speed,
                back_speed=back_speed,
            )

            desired_rotation_curr = mm_utils.desired_rotation_update(
                desired_rotation=self.desired_rotation,
                gamepadstick_left=gamepadstick_left,
                gamepadstick_right=gamepadstick_right,
                camera_azimuth=camera_azimuth,
                desired_strafe=desired_strafe,
                desired_velocity=desired_velocity_curr,
            )

            # Check force search
            self.desired_velocity_change_prev = self.desired_velocity_change_curr.copy()
            self.desired_velocity_change_curr = (desired_velocity_curr - self.desired_velocity) / dt
            self.desired_velocity = desired_velocity_curr.copy()

            self.desired_rotation_change_prev = self.desired_rotation_change_curr.copy()
            self.desired_rotation_change_curr = (
                math.quat_to_scaled_angle_axis(
                    math.quat_abs(math.quat_mul_inv(desired_rotation_curr, self.desired_rotation))
                )
                / dt
            )
            self.desired_rotation = desired_rotation_curr.copy()

            force_search = False
            if self.force_search_timer <= 0.0 and (
                (
                    math.length(self.desired_velocity_change_prev) >= self.desired_velocity_change_threshold
                    and math.length(self.desired_velocity_change_curr) < self.desired_velocity_change_threshold
                )
                or (
                    math.length(self.desired_rotation_change_prev) >= self.desired_rotation_change_threshold
                    and math.length(self.desired_rotation_change_curr) < self.desired_rotation_change_threshold
                )
            ):
                force_search = True
                self.force_search_timer = self.search_time
            elif self.force_search_timer > 0.0:
                self.force_search_timer -= dt

        return force_search, gamepadstick_left, gamepadstick_right

    def _step_predict_trajectory(
        self,
        *,
        dt: float,
        inputs: dict[str, Any],
        camera_azimuth: float,
        desired_strafe: bool,
        fwrd_speed: float,
        side_speed: float,
        back_speed: float,
        is_auto_mode: bool,
        auto_desired_velocity: Optional[np.ndarray],
        gamepadstick_left: np.ndarray,
        gamepadstick_right: np.ndarray,
    ) -> None:
        """Phase 2: predict the desired/rotation/position trajectory windows.

        Mutates the ``self.trajectory_*`` arrays in place.
        """
        if is_auto_mode and auto_desired_velocity is not None:
            turn_rate_rad_s = inputs.get("turn_rate_rad_s", 0.0)
            speed = inputs.get("speed", 0.0)

            for i in range(4):
                t = i * 20.0 * dt
                t_angle_delta = turn_rate_rad_s * t
                t_rot_delta = math.quat_from_angle_axis(t_angle_delta, np.array([0.0, 0.0, 1.0]))
                self.trajectory_desired_rotations[i] = math.quat_mul(t_rot_delta, self.desired_rotation)
                self.trajectory_desired_velocities[i] = (
                    math.quat_mul_vec3(self.trajectory_desired_rotations[i], np.array([1.0, 0.0, 0.0])) * speed
                )

        else:
            # Interactive prediction
            mm_utils.trajectory_desired_rotations_predict(
                desired_rotations=self.trajectory_desired_rotations,
                desired_velocities=self.trajectory_desired_velocities,
                desired_rotation=self.desired_rotation,
                camera_azimuth=camera_azimuth,
                gamepadstick_left=gamepadstick_left,
                gamepadstick_right=gamepadstick_right,
                desired_strafe=desired_strafe,
                dt=20.0 * dt,
            )
            mm_utils.trajectory_desired_velocities_predict(
                desired_velocities=self.trajectory_desired_velocities,
                trajectory_rotations=self.trajectory_rotations,
                desired_velocity=self.desired_velocity,
                camera_azimuth=camera_azimuth,
                gamepadstick_left=gamepadstick_left,
                gamepadstick_right=gamepadstick_right,
                desired_strafe=desired_strafe,
                fwrd_speed=fwrd_speed,
                side_speed=side_speed,
                back_speed=back_speed,
                dt=20.0 * dt,
            )

        mm_utils.trajectory_rotations_predict(
            rotations=self.trajectory_rotations,
            angular_velocities=self.trajectory_angular_velocities,
            rotation=self.simulation_rotation,
            angular_velocity=self.simulation_angular_velocity,
            desired_rotations=self.trajectory_desired_rotations,
            halflife=self.simulation_rotation_halflife,
            dt=20.0 * dt,
        )

        mm_utils.trajectory_positions_predict(
            positions=self.trajectory_positions,
            velocities=self.trajectory_velocities,
            accelerations=self.trajectory_accelerations,
            position=self.simulation_position,
            velocity=self.simulation_velocity,
            acceleration=self.simulation_acceleration,
            desired_velocities=self.trajectory_desired_velocities,
            halflife=self.simulation_velocity_halflife,
            dt=20.0 * dt,
        )

    def _step_build_query(self) -> np.ndarray:
        """Phase 3: assemble the motion-matching query feature vector.

        Combines the current frame's pose-feature prefix with the projected
        trajectory position/direction features.
        """
        query_db = self.db if self.motion_state == "locomotion" else self.traversal_dbs[self.motion_state]
        query_frame_idx = self.frame_idx if self.motion_state == "locomotion" else self.traversal_curr_frame

        query = np.zeros((query_db.nfeatures(),), dtype=np.float32)
        offset = 15
        query[:offset] = (
            query_db.features[query_frame_idx, :offset] * query_db.features_scale[:offset]
            + query_db.features_offset[:offset]
        )

        offset = mm_utils.query_compute_trajectory_position_feature(
            query=query,
            offset=offset,
            root_position=self.bone_positions[0],
            root_rotation=self.bone_rotations[0],
            trajectory_positions=self.trajectory_positions,
        )
        offset = mm_utils.query_compute_trajectory_direction_feature(
            query=query,
            offset=offset,
            root_rotation=self.bone_rotations[0],
            trajectory_rotations=self.trajectory_rotations,
        )
        return query

    def _step_transitions(
        self, query: np.ndarray, trigger_skill: Optional[str], force_search: bool
    ) -> tuple[np.ndarray, bool]:
        """Phase 4: handle locomotion<->traversal transitions.

        On a locomotion->traversal trigger, searches the traversal DB, records
        the matched terrain mesh, and inertializes into the skill clip. On a
        traversal->locomotion exit, adjusts ground height and forces a search.

        Returns the (possibly ground-height-adjusted) ``query`` and the
        (possibly updated) ``force_search`` flag.
        """
        if self.motion_state == "locomotion" and trigger_skill is not None:
            # Locomotion -> Traversal
            self.motion_state = trigger_skill
            self._anchor_foot = None
            traversal_db = self.traversal_dbs[trigger_skill]

            self.start_ground_fit_distance = traversal_db.start_ground_fit_distance

            # Adjust query ground height
            query = mm_utils.adjust_query_ground_height(
                query=query,
                src_ground_height=self.db.search_end_ground_height,
                dst_ground_height=traversal_db.search_start_ground_height,
            )
            # Only bone 1 (base_link in Sim-local frame) carries the
            # platform elevation. Bones 2..30 are local-to-parent joint
            # offsets and must be left as the DB stores them.
            self.curr_bone_positions[1, 2] += (
                -self.db.search_end_ground_height + traversal_db.search_start_ground_height
            )

            best_index, best_cost, _, self.traversal_ending_frame = database_search_fixed(
                best_index=-1,
                best_cost=float("inf"),
                db=traversal_db,
                query=query,
                custom_search_ranges=traversal_db.custom_search_ranges,
            )

            # Terrain matching logic
            mat1_pos = self.bone_positions[0]
            mat1_rot = self.bone_rotations[0]
            mat2_pos = traversal_db.bone_positions[best_index][0]
            mat2_rot = traversal_db.bone_rotations[best_index][0]

            t_rot = math.quat_normalize(math.quat_mul(mat1_rot, math.quat_inv(mat2_rot)))
            t_pos = (mat1_pos - math.quat_mul_vec3(t_rot, mat2_pos)).astype(np.float32)
            t_pos[2] += self.curr_ground_height - traversal_db.search_start_ground_height

            mesh, metadata = traversal_db.find_terrain_mesh(best_index)

            # Transform metadata and mesh
            original_pos = metadata["pos"]
            transformed_pos = t_pos + math.quat_mul_vec3(t_rot, original_pos)
            original_quat = metadata["quat"]
            transformed_quat = math.quat_normalize(math.quat_mul(t_rot, original_quat))

            new_metadata = metadata.copy()
            new_metadata["pos"] = transformed_pos
            new_metadata["quat"] = transformed_quat

            self.terrain_meshes_to_add.append((mesh.copy(), t_pos, t_rot, new_metadata))

            # Transition
            self.trns_bone_positions = traversal_db.bone_positions[best_index].copy()
            self.trns_bone_velocities = traversal_db.bone_velocities[best_index].copy()
            self.trns_bone_rotations = traversal_db.bone_rotations[best_index].copy()
            self.trns_bone_angular_velocities = traversal_db.bone_angular_velocities[best_index].copy()

            mm_utils.inertialize_pose_transition(
                state=self.inertializer,
                root_position=self.bone_positions[0],
                root_velocity=self.bone_velocities[0],
                root_rotation=self.bone_rotations[0],
                root_angular_velocity=self.bone_angular_velocities[0],
                bone_src_positions=self.curr_bone_positions,
                bone_src_velocities=self.curr_bone_velocities,
                bone_src_rotations=self.curr_bone_rotations,
                bone_src_angular_velocities=self.curr_bone_angular_velocities,
                bone_dst_positions=self.trns_bone_positions,
                bone_dst_velocities=self.trns_bone_velocities,
                bone_dst_rotations=self.trns_bone_rotations,
                bone_dst_angular_velocities=self.trns_bone_angular_velocities,
            )
            # transition_dst_position has just absorbed the corrected root;
            # the inertializer will reproduce that position next tick, so
            # zero the standalone offset to avoid double-counting.
            self._root_xy_offset[:] = 0.0

            self.traversal_curr_frame = best_index
            self.traversal_curr_db = traversal_db

        elif (
            self.motion_state in self.traversal_skills and self.traversal_curr_frame >= self.traversal_ending_frame - 1
        ):
            # Traversal -> Locomotion
            query = mm_utils.adjust_query_ground_height(
                query=query,
                src_ground_height=self.traversal_curr_db.search_end_ground_height,
                dst_ground_height=self.db.search_start_ground_height,
            )
            # Only bone 1 (base_link in Sim-local frame) carries the
            # platform elevation. Bones 2..30 are local-to-parent joint
            # offsets and must be left as the DB stores them.
            self.curr_bone_positions[1, 2] += (
                -self.traversal_curr_db.search_end_ground_height + self.db.search_start_ground_height
            )
            self.curr_ground_height += (
                self.traversal_curr_db.search_end_ground_height - self.traversal_curr_db.search_start_ground_height
            )

            self.motion_state = "locomotion"
            self._anchor_foot = None
            force_search = True

        return query, force_search

    def _stop_settled(self) -> bool:
        """Is the robot idle and actually standing on both feet?

        Reads the live inertialized pose, so it answers "has the robot settled",
        not "is the DB frame a standing frame". Feet are compared in root-local
        space: ``dx`` is their fore-aft split and ``dy_sum`` their lateral
        asymmetry, both 0.0 in the standing pose.
        """
        if float(np.linalg.norm(self.desired_velocity)) >= STOP_IDLE_EPS:
            return False

        global_positions, global_velocities, _, _ = forward_kinematics_velocity(
            bone_positions=self.bone_positions,
            bone_velocities=self.bone_velocities,
            bone_rotations=self.bone_rotations,
            bone_angular_velocities=self.bone_angular_velocities,
            bone_parents=self.db.bone_parents,
        )
        if float(np.linalg.norm(global_velocities[bone_left_foot])) >= STOP_FOOT_SPEED_EPS:
            return False
        if float(np.linalg.norm(global_velocities[bone_right_foot])) >= STOP_FOOT_SPEED_EPS:
            return False

        root_position = self.bone_positions[0]
        root_rotation = self.bone_rotations[0]
        left = math.quat_inv_mul_vec3(root_rotation, global_positions[bone_left_foot] - root_position)
        right = math.quat_inv_mul_vec3(root_rotation, global_positions[bone_right_foot] - root_position)
        if abs(float(left[0] - right[0])) >= STOP_STANCE_DX_EPS:
            return False
        if abs(float(left[1] + right[1])) >= STOP_STANCE_DY_EPS:
            return False
        return True

    def _step_search(self, *, dt: float, force_search: bool, query: np.ndarray) -> MotionMatchingDatabase:
        """Phase 5: advance the current clip frame (search or traverse).

        In locomotion, runs the matching search and advances ``frame_idx``; in
        a traversal skill, advances the traversal frame. Refreshes
        ``self.curr_bone_*`` and returns the DB the current frame was read
        from (consumed by phase 6 for ground-height).
        """
        if self.motion_state == "locomotion":
            end_of_anim = (
                mm_utils.database_trajectory_index_clamp(db=self.db, frame=self.frame_idx, offset=1) == self.frame_idx
            )
            best_index = -1 if (end_of_anim or force_search) else self.frame_idx
            best_cost = float("inf")

            # Entering the stop is decided once, then held. The settle test can
            # only be trusted on the way in: the inertializer blend into the
            # standing pose is itself motion, so re-testing right after the snap
            # sees moving feet and would send the search back out to a
            # near-standing frame for a few frames before returning.
            # force_search clears the latch as well as a non-idle command: it
            # marks a deliberate hard transition, such as a skill handing back to
            # locomotion. Without this, a skill that ends while the command is
            # still zero would have its exit pose overwritten by the held stand
            # on the following tick.
            entering_stop = False
            if force_search or float(np.linalg.norm(self.desired_velocity)) >= STOP_IDLE_EPS:
                self.stopped = False
            elif not self.stopped and self._stop_settled():
                self.stopped = True
                entering_stop = True

            if self.stopped and not entering_stop:
                # Already standing: hold the pose. Re-asserted rather than
                # advanced, since advancing would eventually run off the end of
                # the standing clips and force a search. Every frame in 0..627 is
                # the same pose, so there is nothing to advance through anyway.
                self.frame_idx = STOP_FRAME
                self.curr_bone_positions = self.db.bone_positions[STOP_FRAME].copy()
                self.curr_bone_velocities = self.db.bone_velocities[STOP_FRAME].copy()
                self.curr_bone_rotations = self.db.bone_rotations[STOP_FRAME].copy()
                self.curr_bone_angular_velocities = self.db.bone_angular_velocities[STOP_FRAME].copy()
                return self.db

            ran_search = False
            if entering_stop:
                # Cut straight to the standing pose, blending through the same
                # inertializer path a search result would take.
                best_index = STOP_FRAME
            elif force_search or self.search_timer <= 0.0 or end_of_anim:
                ran_search = True
                best_index, best_cost = database_search_fixed(
                    best_index=best_index, best_cost=best_cost, db=self.db, query=query
                )
                # print("best_index: ", best_index, "best_cost: ", best_cost)

            if entering_stop or ran_search:
                if force_search or best_index != self.frame_idx:
                    self.trns_bone_positions = self.db.bone_positions[best_index].copy()
                    self.trns_bone_velocities = self.db.bone_velocities[best_index].copy()
                    self.trns_bone_rotations = self.db.bone_rotations[best_index].copy()
                    self.trns_bone_angular_velocities = self.db.bone_angular_velocities[best_index].copy()

                    mm_utils.inertialize_pose_transition(
                        state=self.inertializer,
                        root_position=self.bone_positions[0],
                        root_velocity=self.bone_velocities[0],
                        root_rotation=self.bone_rotations[0],
                        root_angular_velocity=self.bone_angular_velocities[0],
                        bone_src_positions=self.curr_bone_positions,
                        bone_src_velocities=self.curr_bone_velocities,
                        bone_src_rotations=self.curr_bone_rotations,
                        bone_src_angular_velocities=self.curr_bone_angular_velocities,
                        bone_dst_positions=self.trns_bone_positions,
                        bone_dst_velocities=self.trns_bone_velocities,
                        bone_dst_rotations=self.trns_bone_rotations,
                        bone_dst_angular_velocities=self.trns_bone_angular_velocities,
                    )
                    # See note above: the inertializer absorbed the
                    # corrected root, so zero the standalone offset.
                    self._root_xy_offset[:] = 0.0
                    self.frame_idx = best_index
                self.search_timer = self.search_time

            self.search_timer -= dt
            self.frame_idx += 1

            self.curr_bone_positions = self.db.bone_positions[self.frame_idx].copy()
            self.curr_bone_velocities = self.db.bone_velocities[self.frame_idx].copy()
            self.curr_bone_rotations = self.db.bone_rotations[self.frame_idx].copy()
            self.curr_bone_angular_velocities = self.db.bone_angular_velocities[self.frame_idx].copy()
            curr_db = self.db

        elif self.motion_state in self.traversal_skills:
            self.traversal_curr_frame += 1
            self.curr_bone_positions = self.traversal_curr_db.bone_positions[self.traversal_curr_frame].copy()
            self.curr_bone_velocities = self.traversal_curr_db.bone_velocities[self.traversal_curr_frame].copy()
            self.curr_bone_rotations = self.traversal_curr_db.bone_rotations[self.traversal_curr_frame].copy()
            self.curr_bone_angular_velocities = self.traversal_curr_db.bone_angular_velocities[
                self.traversal_curr_frame
            ].copy()
            curr_db = self.traversal_curr_db

        return curr_db

    def _step_update_inertializer_and_simulation(self, *, dt: float, curr_db: MotionMatchingDatabase) -> None:
        """Phase 6: blend into the current clip pose and update simulation state.

        Runs the pose inertializer (with ground-height compensation on bone 1),
        applies the single-support root-anchor xy offset, and advances the
        simulation position/rotation, finally syncing them to the visualized
        root.
        """
        # Only bone 1 (base_link, in Simulation-local frame) carries the
        # platform elevation. Bones 2..30 are local-to-parent joint offsets
        # whose z must stay as the DB stores them — adding the elevation to
        # every link compounds through FK and corrupts world foot/marker
        # positions.
        self.bone_positions[1, 2] += -self.curr_ground_height + curr_db.search_start_ground_height
        mm_utils.inertialize_pose_update(
            bone_positions=self.bone_positions,
            bone_velocities=self.bone_velocities,
            bone_rotations=self.bone_rotations,
            bone_angular_velocities=self.bone_angular_velocities,
            state=self.inertializer,
            bone_input_positions=self.curr_bone_positions,
            bone_input_velocities=self.curr_bone_velocities,
            bone_input_rotations=self.curr_bone_rotations,
            bone_input_angular_velocities=self.curr_bone_angular_velocities,
            halflife=self.inertialize_blending_halflife,
            dt=dt,
        )
        self.bone_positions[1, 2] -= -self.curr_ground_height + curr_db.search_start_ground_height

        # Single-support root-anchor (freeze model): _root_xy_offset is a
        # persistent xy shift that's always added to bone_positions[0]. It is
        # *updated* during single-support so the supporting foot's world xy
        # tracks the DB clip's foot trajectory (rotated into the visualized
        # yaw); during double-support / flight / non-locomotion the offset is
        # held, so there is no snap at heel-strike or stop. The offset is
        # zeroed at every inertialize_pose_transition, since the inertializer
        # absorbs the corrected root into transition_dst_position there.
        if self.motion_state == "locomotion" and self._anchor_correction_enabled:
            l_contact = bool(self.db.contact_states[self.frame_idx, 0])
            r_contact = bool(self.db.contact_states[self.frame_idx, 1])
            n_contacts = int(l_contact) + int(r_contact)

            if n_contacts == 1:
                support_bone = bone_left_foot if l_contact else bone_right_foot
                snapped = self._anchor_prev_frame_idx >= 0 and abs(self.frame_idx - self._anchor_prev_frame_idx) > 1

                # FK on the *uncorrected* root (before offset is added). The
                # current corrected foot xy is gp_foot_xy + _root_xy_offset.
                gp = forward_kinematics(
                    bone_positions=self.bone_positions,
                    bone_rotations=self.bone_rotations,
                    bone_parents=self.db.bone_parents,
                )
                gp_foot_xy = gp[support_bone, :2].astype(np.float32)

                if self._anchor_foot != support_bone or snapped:
                    # Seed anchor at the current *corrected* foot xy so the
                    # newly-computed offset equals the existing offset on the
                    # seed tick (no jump when single-support resumes).
                    self._anchor_foot = support_bone
                    self._anchor_world_xy = (gp_foot_xy + self._root_xy_offset).copy()
                    self._anchor_prev_db_local_foot = self._db_local_foot(self.frame_idx, support_bone)
                else:
                    # Advance anchor by DB Sim-local delta rotated into world.
                    db_now = self._db_local_foot(self.frame_idx, support_bone)
                    db_delta = db_now - self._anchor_prev_db_local_foot
                    world_delta = math.quat_mul_vec3(self.bone_rotations[0], db_delta)
                    self._anchor_world_xy += world_delta[:2].astype(np.float32)
                    self._anchor_prev_db_local_foot = db_now

                # Solve for the offset that lands foot xy on anchor.
                self._root_xy_offset = (self._anchor_world_xy - gp_foot_xy).astype(np.float32)

                # ~1 Hz diagnostic: max |offset| this second.
                self._anchor_max_dxy = max(self._anchor_max_dxy, float(np.linalg.norm(self._root_xy_offset)))
                now = time.time()
                if now - self._anchor_last_log > 1.0:
                    # print(
                    #     f"[anchor] foot={'L' if support_bone == bone_left_foot else 'R'} "
                    #     f"max|offset|={self._anchor_max_dxy:.4f} m"
                    # )
                    self._anchor_last_log = now
                    self._anchor_max_dxy = 0.0
            else:
                # Double-support / flight: hold the existing offset (no snap).
                self._anchor_foot = None
        else:
            # Non-locomotion or toggle off: still hold the existing offset
            # so disabling mid-walk doesn't snap. The offset will be zeroed
            # at the next inertialize_pose_transition.
            self._anchor_foot = None

        # Always apply the persistent offset, regardless of contact state.
        if np.isfinite(self._root_xy_offset).all():
            self.bone_positions[0, 0] += float(self._root_xy_offset[0])
            self.bone_positions[0, 1] += float(self._root_xy_offset[1])

        self._anchor_prev_frame_idx = self.frame_idx

        self.simulation_position, self.simulation_velocity, self.simulation_acceleration = (
            mm_utils.simulation_positions_update(
                position=self.simulation_position,
                velocity=self.simulation_velocity,
                acceleration=self.simulation_acceleration,
                desired_velocity=self.desired_velocity,
                halflife=self.simulation_velocity_halflife,
                dt=dt,
            )
        )

        self.simulation_rotation, self.simulation_angular_velocity = mm_utils.simulation_rotations_update(
            x=self.simulation_rotation,
            v=self.simulation_angular_velocity,
            x_goal=self.desired_rotation,
            halflife=self.simulation_rotation_halflife,
            dt=dt,
        )

        # Synchronize simulation
        if self.stopped and self.motion_state == "locomotion":
            # While the pose is held there is no search, and the search is what
            # normally closes the heading loop: the trajectory-direction feature
            # in the query encodes the mismatch against the commanded facing, and
            # each pick nudges the body toward it. Held, that loop is open and
            # whatever heading error existed at the moment of the stop is frozen
            # in -- measured at 6-15 deg, against 0-5 deg when the search keeps
            # running.
            #
            # simulation_rotations_update above already produced a rotation
            # damped toward desired_rotation, and the branch below normally
            # discards it. Keep it instead, and re-anchor the inertializer so the
            # held pose follows the new facing.
            #
            # inertialize_pose_update maps the root as
            #     world = transition_dst * inv(transition_src) * input
            # so for a target world rotation the anchor has to be
            #     transition_dst = target * inv(inv(transition_src) * input)
            root_in_transition_frame = math.quat_inv_mul(
                self.inertializer.transition_src_rotation, self.curr_bone_rotations[0]
            )
            self.inertializer.transition_dst_rotation = math.quat_normalize(
                math.quat_mul(self.simulation_rotation, math.quat_inv(root_in_transition_frame))
            )
            self.bone_rotations[0] = self.simulation_rotation.copy()
            self.simulation_position = self.bone_positions[0].copy()
        else:
            synchronized_position = self.bone_positions[0].copy()
            synchronized_rotation = self.bone_rotations[0].copy()
            self.simulation_position = synchronized_position
            self.simulation_rotation = synchronized_rotation

        if self.motion_state != "locomotion":
            self.simulation_velocity = self.bone_velocities[0].copy()
            self.simulation_angular_velocity = self.bone_angular_velocities[0].copy()

    def step(self, dt: float, inputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        # Unpack inputs
        stick_direction = inputs.get("stick_direction", np.zeros(3))
        trigger_skill = inputs.get("trigger_skill", None)
        camera_azimuth = inputs.get("camera_azimuth", 0.0)
        desired_strafe = inputs.get("desired_strafe", False)
        target_speed = inputs.get("target_speed", self.target_speed)
        self.target_speed = target_speed

        # Auto mode inputs
        is_auto_mode = inputs.get("is_auto_mode", False)
        auto_desired_velocity = inputs.get("auto_desired_velocity", None)
        auto_desired_rotation = inputs.get("auto_desired_rotation", None)
        force_search_override = inputs.get("force_search", False)

        # Per-axis target speeds (m/s). The matching DB uses these to pick
        # forward / strafe / backpedal clips with appropriate kinematics.
        fwrd_speed = target_speed * SPEED_AXIS_SCALE_FORWARD
        side_speed = target_speed * SPEED_AXIS_SCALE_SIDE
        back_speed = target_speed * SPEED_AXIS_SCALE_BACK

        # 1. Update Desired Velocity/Rotation
        force_search, gamepadstick_left, gamepadstick_right = self._step_update_desired_velocity(
            dt=dt,
            stick_direction=stick_direction,
            camera_azimuth=camera_azimuth,
            desired_strafe=desired_strafe,
            fwrd_speed=fwrd_speed,
            side_speed=side_speed,
            back_speed=back_speed,
            is_auto_mode=is_auto_mode,
            auto_desired_velocity=auto_desired_velocity,
            auto_desired_rotation=auto_desired_rotation,
            force_search_override=force_search_override,
        )

        # 2. Predict Trajectory
        self._step_predict_trajectory(
            dt=dt,
            inputs=inputs,
            camera_azimuth=camera_azimuth,
            desired_strafe=desired_strafe,
            fwrd_speed=fwrd_speed,
            side_speed=side_speed,
            back_speed=back_speed,
            is_auto_mode=is_auto_mode,
            auto_desired_velocity=auto_desired_velocity,
            gamepadstick_left=gamepadstick_left,
            gamepadstick_right=gamepadstick_right,
        )

        # 3. Build Query
        query = self._step_build_query()

        # 4. Transitions Logic
        query, force_search = self._step_transitions(query, trigger_skill, force_search)

        # 5. Search
        curr_db = self._step_search(dt=dt, force_search=force_search, query=query)

        # 6. Update Inertializer and Simulation
        self._step_update_inertializer_and_simulation(dt=dt, curr_db=curr_db)

        self._update_ground_path()

        return self.bone_positions, self.bone_rotations

    def _update_ground_path(self) -> None:
        # Helper function to create a ground segment
        def create_ground_segment(p1: np.ndarray, p2: np.ndarray, additional_length: float) -> None:
            """Create a ground segment between two points."""
            distance = np.linalg.norm(p2 - p1)
            segment_center_xy = p2 - 0.5 * (distance + additional_length) * (p2 - p1) / distance
            segment_length = max(distance + additional_length, 0.1)  # Minimum length
            segment_width = 0.5
            segment_height = self.curr_ground_height

            # Calculate rotation angle (yaw) for the segment
            direction = p2 - p1
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
                yaw = np.arctan2(direction[1], direction[0])
            else:
                yaw = 0.0
            yaw += np.pi / 2.0
            if yaw > np.pi:
                yaw -= 2 * np.pi

            # Create a rectangular ground segment
            ground_segment = trimesh.creation.box(extents=[segment_width, segment_length, segment_height])

            # Position the segment
            segment_position = np.array(
                [segment_center_xy[0], segment_center_xy[1], self.curr_ground_height - segment_height / 2.0]
            )

            # Create rotation quaternion (rotation around z-axis)
            # viser uses wxyz format
            cos_half_yaw = np.cos(yaw / 2.0)
            sin_half_yaw = np.sin(yaw / 2.0)
            segment_rotation = np.array(
                [
                    cos_half_yaw,  # w
                    0.0,  # x
                    0.0,  # y
                    sin_half_yaw,  # z (rotation around z-axis)
                ]
            )

            # Track terrain mesh for recording
            metadata = {
                "pos": segment_position,
                "quat": segment_rotation,
                "size": np.array([segment_width, segment_length, segment_height]),
            }

            self.terrain_meshes_to_add.append(
                (ground_segment.copy(), segment_position.copy(), segment_rotation.copy(), metadata)
            )

        # Draw terrain meshes for higher ground height
        # Create ground path when in locomotion mode with elevated ground
        if self.motion_state == "locomotion" and self.curr_ground_height > 0.05:
            import trimesh

            # Get current character root position (x, y)
            root_pos_xy = self.bone_positions[0][:2]  # Only x, y coordinates

            # Check if we should add a new path point
            if self.last_ground_path_position is not None:
                distance = np.linalg.norm(root_pos_xy - self.last_ground_path_position)
                if len(self.ground_path_positions) == 0:
                    if distance >= self.start_ground_fit_distance:
                        self.ground_path_positions.append(root_pos_xy.copy())
                        self.last_ground_path_position = root_pos_xy.copy()
                else:
                    if distance >= self.ground_path_segment_size:
                        additional_length = 0.3 if len(self.ground_path_positions) >= 2 else 0.0
                        create_ground_segment(
                            p1=self.last_ground_path_position, p2=root_pos_xy, additional_length=additional_length
                        )
                        self.ground_path_positions.append(root_pos_xy.copy())
                        self.last_ground_path_position = root_pos_xy.copy()
            else:
                self.last_ground_path_position = root_pos_xy.copy()

        else:
            self.ground_path_positions = []
            self.last_ground_path_position = None


def generate_segment(
    sys: "MotionMatchingSystem",
    speed: float,
    turn_angle_degrees: float,
    duration_s: float,
    vel_cmd_idx: int,
    dt: float = 1.0 / 60.0,
    force_search_first_frame: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = int(duration_s / dt)
    qpos_list = []
    vel_cmd_list = []
    vel_cmd_dim = VEL_CMD_DIM

    turn_rate_rad_s = np.deg2rad(turn_angle_degrees) / duration_s

    for step_idx in range(num_steps):
        # Update desired rotation externally for auto mode
        angle_delta = turn_rate_rad_s * dt
        rot_delta = math.quat_from_angle_axis(angle_delta, np.array([0.0, 0.0, 1.0]))

        sys.desired_rotation = math.quat_mul(rot_delta, sys.desired_rotation)

        forward_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sys.desired_velocity = math.quat_mul_vec3(sys.desired_rotation, forward_local) * speed

        inputs = {
            "is_auto_mode": True,
            "auto_desired_velocity": sys.desired_velocity,
            "auto_desired_rotation": sys.desired_rotation,
            "force_search": (force_search_first_frame and step_idx == 0),
            "turn_rate_rad_s": turn_rate_rad_s,
            "speed": speed,
        }

        bone_positions, bone_rotations = sys.step(dt, inputs)

        # Capture qpos
        dof_rotations = bone_rotations[-29:]
        dof_aa = math.quat_to_scaled_angle_axis(dof_rotations)
        dof = np.sum(dof_aa * joint_axes, axis=-1)

        global_positions, global_rotations = forward_kinematics(
            bone_positions=bone_positions,
            bone_rotations=bone_rotations,
            bone_parents=sys.db.bone_parents,
            return_rotations=True,
        )

        qpos = np.concatenate([global_rotations[1], global_positions[1], dof], dtype=np.float32)
        qpos_list.append(qpos)

        vel_cmd = np.zeros(vel_cmd_dim)
        vel_cmd[vel_cmd_idx] = 1.0
        vel_cmd_list.append(vel_cmd)

    return np.array(qpos_list), np.array(vel_cmd_list)


def generate_skill_sequence(
    sys: "MotionMatchingSystem",
    skill_name: str,
    vel_cmd_idx: int,
    dt: float = 1.0 / 60.0,
    speed: float = 1.0,
    angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    qpos_list = []
    vel_cmd_list = []
    vel_cmd_dim = VEL_CMD_DIM

    # Keep "walking straight at the commanded heading" active for the whole
    # skill, so velocity does not decay and the hand-off back to locomotion is
    # smooth.
    #
    # Driven through auto mode, not the interactive path, so that
    # desired_rotation stays exactly what the scenario asked for. The
    # interactive path recomputes it every tick as
    #     yaw(desired_velocity), where
    #     desired_velocity = simulation_rotation * ([fwrd, side, 0] *
    #                        inv(simulation_rotation) * stick)
    # and because the per-axis scales differ (SPEED_AXIS_SCALE_SIDE is 0.75,
    # forward is 1.0) that round trip is not an identity: it pulls the commanded
    # heading toward whatever the body currently faces, by
    # ``atan2(0.75 sin d, cos d)`` where ``d`` is the angle between them. A skill
    # exited with the body 26 deg off command moved desired_rotation 6.1 deg,
    # while the same skill exited 10 deg off moved it 2.7 deg -- so the scenario's
    # heading command ended up depending on how far the body had drifted, and the
    # post-skill search saw a different query for the same scripted command.
    forward_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    desired_rotation = sys.desired_rotation.copy()
    forward_world = math.quat_mul_vec3(desired_rotation, forward_local)
    auto_desired_velocity = (forward_world * speed).astype(np.float32)

    # 1. Trigger the skill
    inputs = {
        "trigger_skill": skill_name,
        "is_auto_mode": True,
        "auto_desired_velocity": auto_desired_velocity,
        "auto_desired_rotation": desired_rotation,
        "target_speed": speed,
    }

    # We need to capture at least one frame where the trigger happens
    # This might transition immediately to the skill
    bone_positions, bone_rotations = sys.step(dt, inputs)

    def capture_frame(b_pos: np.ndarray, b_rot: np.ndarray) -> None:
        dof_rotations = b_rot[-29:]
        dof_aa = math.quat_to_scaled_angle_axis(dof_rotations)
        dof = np.sum(dof_aa * joint_axes, axis=-1)

        global_positions, global_rotations = forward_kinematics(
            bone_positions=b_pos, bone_rotations=b_rot, bone_parents=sys.db.bone_parents, return_rotations=True
        )

        qpos = np.concatenate([global_rotations[1], global_positions[1], dof], dtype=np.float32)
        qpos_list.append(qpos)

        vel_cmd = np.zeros(vel_cmd_dim)
        vel_cmd[vel_cmd_idx] = 1.0
        vel_cmd_list.append(vel_cmd)

    capture_frame(bone_positions, bone_rotations)

    # Continue until skill ends
    # sys.step handles the transition logic.
    # If transition successful, motion_state is now skill_name.
    # It automatically switches back to 'locomotion' when done.

    while sys.motion_state == skill_name:
        # Same constant command every tick, as if a key were held. Re-deriving it
        # from sys.desired_rotation would drift; under auto mode desired_rotation
        # cannot move anyway, so the two now agree by construction.
        inputs = {
            "trigger_skill": None,
            "is_auto_mode": True,
            "auto_desired_velocity": auto_desired_velocity,
            "auto_desired_rotation": desired_rotation,
            "target_speed": speed,
        }
        bone_positions, bone_rotations = sys.step(dt, inputs)
        capture_frame(bone_positions, bone_rotations)

    return np.array(qpos_list), np.array(vel_cmd_list)


_worker_sys = None
# Set by run_generation BEFORE the pool is constructed so spawned workers
# inherit the value via module-import (workers re-import this module on spawn).
_worker_required_skills = None


def init_worker(required_skills: Optional[Any] = None) -> None:
    """Pool initializer: build the per-worker :class:`MotionMatchingSystem`.

    Args:
        required_skills: Optional iterable of skill names. If supplied, only
            those skills + locomotion are loaded; saves significant memory and
            startup time when a scenario only needs a few skills. Defaults to
            ``_worker_required_skills`` if unset (so the pool dispatcher can
            stash the value before creating the pool).
    """
    global _worker_sys
    if required_skills is None:
        required_skills = _worker_required_skills
    try:
        _worker_sys = MotionMatchingSystem(required_skills=required_skills)
    except Exception as e:
        print(f"Error initializing worker: {e}")
        raise e


def process_scenario(scenario: dict[str, Any], folder: str, force: bool, speed_up_ratio: float) -> None:
    global _worker_sys, num_terrain_variants
    sys = _worker_sys

    name = scenario["name"]
    filename = f"{name}_motion.npz"

    output_file = os.path.join(folder, filename)

    if os.path.exists(output_file) and not force:
        print(f"Skipping {name} because it already exists")
        return

    sys.reset()
    sys.terrain_meshes_to_add = []  # Clear terrain meshes

    print(f"Generating {name}...")

    all_qpos = []
    all_vel_cmd = []

    for step in scenario["steps"]:
        step_type = step["type"]

        if step_type == "segment" or step_type == "locomotion":
            speed = step["speed"]
            angle = step["angle"]
            duration = step["duration"]
            vel_cmd_idx = step["vel_cmd_idx"]
            force_search = step.get("force_search", False)

            # print(f"  Generating segment: speed={speed}, angle={angle}, duration={duration}")
            qpos, vel = generate_segment(
                sys=sys,
                speed=speed,
                turn_angle_degrees=angle,
                duration_s=duration,
                vel_cmd_idx=vel_cmd_idx,
                force_search_first_frame=force_search,
            )
            all_qpos.append(qpos)
            all_vel_cmd.append(vel)

        elif step_type == "skill":
            skill_name = step["skill_name"]  # e.g. "step"
            vel_cmd_idx = step["vel_cmd_idx"]
            speed = step.get("speed", 1.0)  # Default to 1.0 if not specified
            angle = step.get("angle", 0.0)  # Default to 0.0 if not specified

            # print(f"  Generating skill: {skill_name} with speed={speed}, angle={angle}")
            qpos, vel = generate_skill_sequence(
                sys=sys, skill_name=skill_name, vel_cmd_idx=vel_cmd_idx, speed=speed, angle=angle
            )
            all_qpos.append(qpos)
            all_vel_cmd.append(vel)

    if not all_qpos:
        print(f"  No segments generated for {name}.")
        return

    qpos = np.concatenate(all_qpos, axis=0)
    vel_cmd = np.concatenate(all_vel_cmd, axis=0)

    required_fps = 50
    qpos = adjust_data_speed(qpos, speed_up_ratio)
    vel_cmd = adjust_vel_cmd_speed(vel_cmd, speed_up_ratio)
    vel_cmd = resample_vel_cmd(vel_cmd, required_fps, original_fps=60)

    # Some locomotion scenarios request nearby clutter after their final root
    # trajectory is known. This terrain is deliberately post-processed: it must
    # not influence which motion-matching frames are selected.
    terrain_rng = None
    offpath_variant_cfg = None
    offpath_cfg = scenario.get("offpath_obstacles")
    if offpath_cfg is not None:
        offpath_cfg = dict(offpath_cfg)
        terrain_rng = np.random.default_rng(offpath_cfg.pop("seed", None))
        min_size = offpath_cfg.get("min_size", (0.25, 0.25, 0.20))
        max_size = offpath_cfg.get("max_size", (0.65, 0.65, 1.00))
        offpath_variant_cfg = {
            "path_clearance": offpath_cfg["path_clearance"],
            "min_center_distance": offpath_cfg.get("min_center_distance"),
            "max_path_distance": offpath_cfg["max_path_distance"],
            "min_size": min_size,
            "max_size": max_size,
            "obstacle_gap": offpath_cfg.get("obstacle_gap", 0.05),
        }
        # The reference OBJ and every NPY scene use the same maximum footprint.
        # No local-jitter reserve is needed because variants are fresh layouts.
        offpath_cfg["variant_max_half_extent_x"] = None
        offpath_cfg["position_jitter"] = 0.0
        offpath_obstacles = sample_offpath_box_obstacles(
            qpos[:, 4:6],
            rng=terrain_rng,
            **offpath_cfg,
        )
        sys.terrain_meshes_to_add.extend(offpath_obstacles)
        print(f"Added {len(offpath_obstacles)} off-path box obstacles")

    save_motion_npz(qpos, output_file, target_fps=required_fps, source_fps=60, vel_cmd=vel_cmd)
    print(f"Saved to {output_file}")

    # Save terrain
    posed_meshes = [(mesh, pos, rot) for mesh, pos, rot, _ in sys.terrain_meshes_to_add]
    combined_mesh = build_combined_terrain_mesh(posed_meshes)
    mesh_filename = output_file.replace("_motion.npz", "_terrain.obj")
    combined_mesh.export(mesh_filename)

    # Save terrain npy
    terrain_metadata = [metadata for _, _, _, metadata in sys.terrain_meshes_to_add]
    if offpath_variant_cfg is not None:
        random_terrains = randomize_offpath_terrain_variants(
            terrain_metadata,
            qpos[:, 4:6],
            num_variants=num_terrain_variants,
            rng=terrain_rng,
            **offpath_variant_cfg,
        )
    else:
        random_terrains = randomize_terrain_variants(
            terrain_metadata,
            num_variants=num_terrain_variants,
            rng=terrain_rng,
        )
    npy_file = output_file.replace("_motion.npz", "_terrain.npy")
    np.save(npy_file, random_terrains)


def run_generation(args: argparse.Namespace) -> None:
    # sys = MotionMatchingSystem() # Instantiated in workers

    folder = os.path.join("motion_matching_results", args.scenario)
    if args.force:
        import shutil

        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

    scenarios = []
    try:
        import motion_matching.scenarios as cfg_script

        if hasattr(cfg_script, args.scenario):
            scenarios, speed_up_ratio = getattr(cfg_script, args.scenario)()
            print(f"Loaded scenario '{args.scenario}' from motion_matching.scenarios")
        else:
            print(f"Error: Scenario function '{args.scenario}' not found in motion_matching.scenarios")
            return
    except ImportError:
        import sys
        import traceback

        print("Error importing motion_matching.scenarios:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return

    print(f"Generating trajectories based on {len(scenarios)} scenarios...")

    # Identify the subset of skills this scenario actually dispatches; workers
    # will only load these (plus locomotion). Saves ~25 of 28 DBs per worker
    # for typical scenarios.
    required_skills = collect_required_skills(scenarios)
    print(f"Required skills for this scenario: {sorted(required_skills) or '(none, locomotion-only)'}")

    # Prepare tasks
    tasks = [(scenario, folder, args.force, speed_up_ratio) for scenario in scenarios]

    # Determine number of jobs. Cap at the task count — spinning up 8 spawn
    # workers each loading a subset of databases is wasted work for a 1-config
    # scenario. Anything <= 1 task runs in-process (avoids pool overhead).
    requested_jobs = args.jobs if args.jobs > 0 else multiprocessing.cpu_count()
    num_jobs = min(requested_jobs, len(tasks))

    if num_jobs <= 1:
        print(f"Running in-process ({len(tasks)} task(s), no pool overhead).")
        init_worker(required_skills=required_skills)  # populates module-level _worker_sys
        for task in tasks:
            process_scenario(*task)
    else:
        print(f"Starting parallel generation with {num_jobs} jobs.")
        # Use spawn context for safety with libraries like numpy/viser/trimesh
        # that hold internal locks; fork would be faster but unsafe on macOS.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_jobs, initializer=init_worker, initargs=(required_skills,)) as pool:
            pool.starmap(process_scenario, tasks)

    print("Generation complete.")


def run_interactive(args: argparse.Namespace) -> None:
    import viser

    from motion_matching.ui.gui_controller import (
        FileKeyboardController,
        keyboard_skills_mapping,
        speed_modifier_keys,
    )
    from motion_matching.viz import add_world_scene_with_g1

    sys = MotionMatchingSystem()

    sys.reset()

    #########################################################
    # Initialize viser server
    #########################################################
    server = viser.ViserServer(port=8081, verbose=False)

    with server.gui.add_folder("Controls"):
        mode = server.gui.add_dropdown("Mode", options=["Manual", "MotionMatching"], initial_value="MotionMatching")
        auto_play = server.gui.add_checkbox("Auto Play", False)
        speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        frame_slider = server.gui.add_slider("Frame", min=0, max=sys.db.nframes() - 1, step=1, initial_value=0)

        # Keyboard control instructions
        server.gui.add_markdown("Keyboard Control (IKJL + Space)")
        server.gui.add_markdown("I: Forward, K: Backward, J: Left, L: Right, Space: Space")
        server.gui.add_markdown("Use the pop-up GUI window to control the keyboard")
        server.gui.add_markdown("The GUI window will display the key status")
        server.gui.add_markdown(
            f"Tab: toggle Low ({LOCOMOTION_LOW_SPEED} m/s) / High ({LOCOMOTION_HIGH_SPEED} m/s) speed"
        )

        # Debug visualization controls
        show_trajectory = server.gui.add_checkbox("Show Predicted Trajectory", True)
        show_root_projection = server.gui.add_checkbox("Show Root XY Projection", True)

        # Trajectory prediction halflife sliders (shorter = more responsive, longer = smoother).
        # Both start at the class default (0.27) and are not touched by the speed
        # toggle, so interactive matches generate mode.
        velocity_halflife_slider = server.gui.add_slider(
            "Velocity Halflife", min=0.01, max=0.5, step=0.01, initial_value=sys.simulation_velocity_halflife
        )
        rotation_halflife_slider = server.gui.add_slider(
            "Rotation Halflife", min=0.01, max=0.5, step=0.01, initial_value=sys.simulation_rotation_halflife
        )

        @velocity_halflife_slider.on_update
        def _(_: Any) -> None:
            sys.simulation_velocity_halflife = float(velocity_halflife_slider.value)

        @rotation_halflife_slider.on_update
        def _(_: Any) -> None:
            sys.simulation_rotation_halflife = float(rotation_halflife_slider.value)

        # Camera follow control
        server.gui.add_markdown("---")
        server.gui.add_markdown("Camera Follow")
        camera_follow = server.gui.add_checkbox("Follow Character", True)
        camera_offset_x = server.gui.add_slider("Camera Offset X", min=-5.0, max=5.0, step=0.1, initial_value=-2)
        camera_offset_y = server.gui.add_slider("Camera Offset Y", min=-5.0, max=5.0, step=0.1, initial_value=1.0)
        camera_offset_z = server.gui.add_slider("Camera Offset Z", min=-5.0, max=5.0, step=0.1, initial_value=1.0)
        camera_smoothness = server.gui.add_slider("Camera Smoothness", min=0.01, max=0.5, step=0.01, initial_value=0.05)
        camera_speed_follow = server.gui.add_checkbox("Follow Speed", True)

        # Reset to the configured static startup pose.
        server.gui.add_markdown("---")
        reset_pose_button = server.gui.add_button("Reset to Stiff Startup Pose")

        # Foot contact label readout (left/right foot, from DB contact_states)
        server.gui.add_markdown("---")
        contact_label_text = server.gui.add_text("Contact (L/R)", initial_value="L: -  R: -")
        show_contact_markers = server.gui.add_checkbox("Show Contact Markers", False)
        foot_anchor_toggle = server.gui.add_checkbox("Foot Anchor Correction", True)
        anchor_foot_text = server.gui.add_text("Anchor Foot", initial_value="-")

        @foot_anchor_toggle.on_update
        def _(_: Any) -> None:
            sys._anchor_correction_enabled = bool(foot_anchor_toggle.value)

        # Recording controls
        server.gui.add_markdown("---")
        server.gui.add_markdown("Motion Recording")
        start_recording = server.gui.add_button("Start Recording")
        end_recording = server.gui.add_button("End Recording")
        recording_status = server.gui.add_text("Recording: OFF", initial_value="Recording: OFF")

    # World origin + grid + robot-root frame + G1 URDF visualizer.
    root_frame, viser_urdf = add_world_scene_with_g1(server)
    print("Loaded G1 URDF for visualization")

    # Create file communication keyboard controller
    keyboard = FileKeyboardController()
    if not keyboard.start():
        print("The keyboard controller has failed to start, exiting the program")
        return

    fps = 60.0
    dt = 1.0 / fps
    current_frame = 0
    terrain_cnt = 0

    camera_azimuth = 0.0
    desired_strafe = False
    # Per-client snap counter: while > 0, force camera to target without lerp.
    # Reset on each connect (covers page refresh) so the lerp can't pull the
    # camera in from the browser's stale default position.
    camera_snap_ticks: dict[int, int] = {}
    CAMERA_SNAP_FRAMES = 30

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        camera_snap_ticks[client.client_id] = CAMERA_SNAP_FRAMES

    @server.on_client_disconnect
    def _(client: viser.ClientHandle) -> None:
        camera_snap_ticks.pop(client.client_id, None)

    speed_state = "low"  # toggled by speed_modifier_keys (e.g. Tab)
    HIGH_SPEED_FAMILIES = ("step", "climb_76", "climb_94")
    prev_speed_toggle_pressed = False

    # Recording state
    is_recording = False
    recorded_frames = []  # List to store recorded data: [root_rot (4), root_pos (3), dof (29)]
    terrain_meshes = {}  # Dictionary to store terrain meshes: {name: (mesh, position, rotation)}

    # Last printed contact label, so we only log on change.
    prev_contact_label = None

    # Initialize URDF with zero DOF (initial pose)
    initial_dof = np.zeros(29, dtype=np.float32)
    viser_urdf.update_cfg(initial_dof)

    @reset_pose_button.on_click
    def _(_: Any) -> None:
        nonlocal terrain_cnt
        # Back to the DB's first frame. The system stays frozen on it until
        # the user provides a velocity command.
        sys.reset()
        # Clear any terrain meshes added since the last reset so visuals
        # match the freshly reset state. Viser's API is ``remove_by_name``
        # (not ``remove``); the latter silently no-ops.
        for terrain_name in list(terrain_meshes.keys()):
            try:
                server.scene.remove_by_name(terrain_name)
            except Exception as exc:
                print(f"Failed to remove {terrain_name}: {exc}")
        terrain_meshes.clear()
        terrain_cnt = 0
        # Snap each connected client's camera straight to character_pos +
        # offset on the next render tick (skip the smoothing lerp), so the
        # view jumps to the reset pose instead of slowly drifting in.
        for client_id in server.get_clients().keys():
            camera_snap_ticks[client_id] = CAMERA_SNAP_FRAMES
        print("Reset to stiff startup pose")

    # Button handlers for recording
    @start_recording.on_click
    def _(_: Any) -> None:
        nonlocal is_recording, recorded_frames, terrain_meshes
        if not is_recording:
            is_recording = True
            recorded_frames = []
            # Don't clear terrain_meshes - we want to save all terrain meshes that exist
            recording_status.value = "Recording: ON"
            print("Recording started")

    @end_recording.on_click
    def _(_: Any) -> None:
        nonlocal is_recording, recorded_frames, terrain_meshes
        if is_recording:
            is_recording = False
            recording_status.value = "Recording: OFF"

            if len(recorded_frames) > 0:
                # The terrain helpers import trimesh internally; fail early with a
                # friendly message if it isn't installed.
                import importlib.util

                if importlib.util.find_spec("trimesh") is None:
                    print("Error: trimesh library is required to save terrain meshes. Please install it via pip.")
                    return

                # Generate timestamp and create folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = f"motion_matching_results/mm{timestamp[:8]}/motion_{timestamp}"
                os.makedirs(folder_name, exist_ok=True)

                # Convert to numpy array: (N, 36) where N is number of frames
                recorded_data = np.array(recorded_frames, dtype=np.float32)

                required_fps = 50
                # Save motion file (re-sampled from capture fps to 50hz, FAR-tracking schema)
                motion_filename = os.path.join(folder_name, "..", f"motion_{timestamp}.npz")
                save_motion_npz(recorded_data, motion_filename, target_fps=required_fps, source_fps=fps)
                print(f"Motion saved to {motion_filename} ({len(recorded_frames)} frames)")

                # Collect posed terrain meshes + box metadata in stable name order.
                posed_meshes = []
                metadata_list = []
                for terrain_name in sorted(terrain_meshes.keys()):
                    if terrain_name.startswith("/terrain_mesh/"):
                        mesh, mesh_position, mesh_rotation, metadata = terrain_meshes[terrain_name]
                        posed_meshes.append((mesh, mesh_position, mesh_rotation))
                        metadata_list.append(metadata)

                if metadata_list:
                    # Save random terrains to a .npy file
                    random_terrains = randomize_terrain_variants(metadata_list, num_variants=num_terrain_variants)
                    random_terrains_filename = os.path.join(folder_name, "terrain.npy")
                    np.save(random_terrains_filename, random_terrains)
                    print(f"Random terrains saved to {random_terrains_filename} ({random_terrains.shape[0]} terrains)")

                # Combine obstacle meshes + ground plane and export as OBJ.
                combined_mesh = build_combined_terrain_mesh(posed_meshes)
                mesh_filename = os.path.join(folder_name, "multi_boxes_scaled.obj")
                combined_mesh.export(mesh_filename)
                print(f"Terrain mesh saved to {mesh_filename} ({len(posed_meshes)} terrain meshes + ground plane)")

                recorded_frames = []
            else:
                print("No frames recorded")

    previous_time = time.time()

    #########################################################
    # Start motion matching
    #########################################################
    while True:
        # frame_counter += 1

        # Update keyboard state
        keyboard.update()

        # Check if we should exit
        if not keyboard.running:
            print("The keyboard controller has stopped, exiting the program")
            break

        # Unit-direction stick from the keyboard (no magnitude — speed is set
        # by the speed toggle below).
        stick_direction = keyboard.get_desired_velocity()

        # Toggle speed state (rising edge on Tab)
        speed_toggle_pressed = any(keyboard.is_key_pressed(k) for k in speed_modifier_keys)
        if speed_toggle_pressed and not prev_speed_toggle_pressed:
            speed_state = "high" if speed_state == "low" else "low"
            print(f"[speed] toggled to {speed_state}")
        prev_speed_toggle_pressed = speed_toggle_pressed
        target_speed = LOCOMOTION_HIGH_SPEED if speed_state == "high" else LOCOMOTION_LOW_SPEED

        trigger_skill = None
        for key, skill in keyboard_skills_mapping.items():
            if keyboard.is_key_pressed(key):
                trigger_skill = skill
                # Apply speed modifier: insert "_high_speed" between family and direction
                if speed_state == "high":
                    for family in HIGH_SPEED_FAMILIES:
                        if skill == family or skill.startswith(family + "_"):
                            suffix = skill[len(family) :]  # '' or '_up' or '_down'
                            candidate = family + "_high_speed" + suffix
                            if candidate in sys.traversal_dbs:
                                trigger_skill = candidate
                            break
                break

        final_bone_positions = None
        final_bone_rotations = None

        if mode.value == "MotionMatching":
            inputs = {
                "stick_direction": stick_direction,
                "trigger_skill": trigger_skill,
                "camera_azimuth": camera_azimuth,
                "desired_strafe": desired_strafe,
                "target_speed": target_speed,
            }

            # Step the system
            final_bone_positions, final_bone_rotations = sys.step(dt, inputs)

            # Update local terrain meshes if new ones were added by sys
            while sys.terrain_meshes_to_add:
                mesh, pos, rot, metadata = sys.terrain_meshes_to_add.pop(0)

                terrain_name = f"/terrain_mesh/{terrain_cnt}"
                server.scene.add_mesh_trimesh(terrain_name, mesh=mesh, position=pos, wxyz=rot)

                terrain_meshes[terrain_name] = (mesh, pos, rot, metadata)
                terrain_cnt += 1

        else:
            # Manual mode - use database directly
            # Advance time (Manual autoplay)
            if auto_play.value:
                current_frame += dt * fps * speed.value

            # Clamp and sync GUI
            if current_frame >= sys.db.nframes():
                current_frame = 0.0
            if current_frame < 0.0:
                current_frame = float(sys.db.nframes() - 1)
            frame_slider.value = int(current_frame)
            next_frame = int(current_frame)

            current_frame = float(frame_slider.value)

            final_bone_positions = sys.db.bone_positions[next_frame]
            final_bone_rotations = sys.db.bone_rotations[next_frame]

        #########################################################
        # Visualization and Recording
        #########################################################

        # Foot contact label from the active DB at the current frame.
        # contact_states is (N, 2) of bools: [left_foot, right_foot].
        if mode.value == "MotionMatching":
            if sys.motion_state == "locomotion":
                contact_db = sys.db
                contact_frame = sys.frame_idx
            else:
                contact_db = sys.traversal_curr_db
                contact_frame = sys.traversal_curr_frame
        else:
            contact_db = sys.db
            contact_frame = next_frame
        l_contact, r_contact = contact_db.contact_states[contact_frame]
        contact_label = f"L: {int(l_contact)}  R: {int(r_contact)}"
        contact_label_text.value = contact_label
        if contact_label != prev_contact_label:
            # print(f"[contact] frame={contact_frame} {contact_label}")
            prev_contact_label = contact_label

        anchor_foot_text.value = (
            "L" if sys._anchor_foot == bone_left_foot else "R" if sys._anchor_foot == bone_right_foot else "-"
        )

        if mode.value == "MotionMatching":
            # Draw predicted trajectory (orange)
            if show_trajectory.value:
                ORANGE = np.array([255, 165, 0])
                mm_utils.draw_trajectory(
                    server=server,
                    trajectory_positions=sys.trajectory_positions,
                    trajectory_rotations=sys.trajectory_rotations,
                    color=ORANGE,
                    name_prefix="predicted_trajectory",
                )
            else:
                # Clear trajectory visualization when disabled
                for i in range(1, 4):  # trajectory has 4 points, skip index 0
                    try:
                        server.scene.remove(f"/predicted_trajectory_point_{i}")
                        server.scene.remove(f"/predicted_trajectory_dir_{i}")
                        if i > 1:
                            server.scene.remove(f"/predicted_trajectory_path_{i}")
                    except Exception:
                        pass  # Ignore if nodes don't exist

            # Draw root XY projection (red point on ground plane)
            if show_root_projection.value:
                # Get root position (bone_positions[0])
                root_pos = final_bone_positions[0]
                # Project to XY plane (set z to 0)
                root_projection = np.array([root_pos[0], root_pos[1], 0.0])

                # Draw red sphere at projection point
                server.scene.add_icosphere(
                    "/root_projection_point",
                    position=root_projection,
                    radius=0.08,
                    color=np.array([255, 0, 0]),  # Red color
                    subdivisions=2,
                )

                # Draw line from root to projection (vertical line)
                server.scene.add_line_segments(
                    "/root_projection_line",
                    points=np.array([[root_pos, root_projection]]),
                    colors=np.array([[np.array([255, 0, 0]), np.array([255, 0, 0])]]),
                    line_width=2.0,
                )
            else:
                # Clear projection visualization when disabled
                try:
                    server.scene.remove("/root_projection_point")
                    server.scene.remove("/root_projection_line")
                except Exception:
                    pass  # Ignore if nodes don't exist

        # Draw skeleton using URDF meshes, default not visible
        try:
            # Translate final_bone_rotations to dof
            dof_rotations = final_bone_rotations[-29:]
            # assert dof_rotations.shape[0] == len(joint_axes)

            dof_aa = math.quat_to_scaled_angle_axis(dof_rotations)
            dof = np.sum(dof_aa * joint_axes, axis=-1)

            global_positions, global_rotations = forward_kinematics(
                bone_positions=final_bone_positions,
                bone_rotations=final_bone_rotations,
                bone_parents=sys.db.bone_parents,
                return_rotations=True,
            )

            # Update robot root frame position and rotation
            root_frame.position = global_positions[1]
            root_frame.wxyz = global_rotations[1]

            # Update URDF with the calculated DOF values
            viser_urdf.update_cfg(dof)

            # Blue spheres at feet that are currently in contact (DB label).
            BLUE = np.array([0, 100, 255])
            markers_visible = show_contact_markers.value
            for side, in_contact, bone in (
                ("left", bool(l_contact), bone_left_foot),
                ("right", bool(r_contact), bone_right_foot),
            ):
                node = f"/contact_marker/{side}"
                if markers_visible and in_contact:
                    server.scene.add_icosphere(
                        node, position=global_positions[bone], radius=0.06, color=BLUE, subdivisions=2
                    )
                else:
                    try:
                        server.scene.remove_by_name(node)
                    except Exception:
                        pass

            # Record motion data if recording is active
            if is_recording:
                # Format: [root_rot (wxyz, 4), root_pos (xyz, 3), dof (29)] = 36 dims
                root_rot = global_rotations[1]  # wxyz quaternion
                root_pos = global_positions[1]  # xyz position
                frame_data = np.concatenate([root_rot, root_pos, dof], dtype=np.float32)
                recorded_frames.append(frame_data)

            # Build line segments
            bone_lines = []
            colors = []
            for child_idx, parent_idx in enumerate(sys.db.bone_parents):
                if parent_idx == 0:
                    continue
                elif parent_idx != -1:
                    p = global_positions[parent_idx]
                    c = global_positions[child_idx]
                    if not (np.isfinite(p).all() and np.isfinite(c).all()):
                        continue
                    bone_lines.append([p, c])
                    colors.append([[100, 255, 100], [100, 255, 100]])

            server.scene.add_line_segments(
                "/skeleton", points=np.array(bone_lines), colors=np.array(colors), line_width=4.0, visible=False
            )

            # Get root position for camera follow
            root = global_positions[1] if len(global_positions) > 1 else np.array([0.0, 0.0, 0.0])

            # Camera follow logic
            if camera_follow.value:
                # Get all connected clients
                clients = server.get_clients()
                for client in clients.values():
                    # Use the root position (Hips) as the follow target
                    character_pos = root

                    # Calculate camera offset (relative to character orientation)
                    offset_x = camera_offset_x.value
                    offset_y = camera_offset_y.value
                    offset_z = camera_offset_z.value

                    # If speed follow is enabled, adjust camera distance based on character speed
                    if camera_speed_follow.value and mode.value == "MotionMatching":
                        # Calculate character current speed
                        character_speed = np.linalg.norm(sys.simulation_velocity)
                        # Adjust camera distance based on speed (faster speed, longer distance)
                        speed_factor = 1.0 + min(character_speed * 0.1, 1.0)
                        offset_z *= speed_factor

                    # Calculate camera position (behind character, relative to character orientation)
                    # camera_offset = offset_x * character_right + offset_y * np.array([0.0, 1.0, 0.0]) + offset_z * character_forward
                    camera_offset = np.array([offset_x, offset_y, offset_z])
                    camera_pos = character_pos + camera_offset

                    # Set camera to look at character
                    client.camera.look_at = character_pos + np.array([0.0, 1.0, 0.0])
                    client.camera.up_direction = np.array([0.0, 0.0, 1.0])

                    # Smoothly update camera position (avoid sudden jump)
                    current_pos = client.camera.position
                    snap_remaining = camera_snap_ticks.get(client.client_id, 0)
                    if snap_remaining > 0 or not np.isfinite(current_pos).all():
                        client.camera.position = camera_pos
                        if snap_remaining > 0:
                            camera_snap_ticks[client.client_id] = snap_remaining - 1
                    else:
                        alpha = camera_smoothness.value
                        smooth_pos = alpha * camera_pos + (1 - alpha) * current_pos
                        client.camera.position = smooth_pos

        except Exception:
            pass

        #########################################################
        # Sleep to ensure correct frame rate
        #########################################################
        current_time = time.time()
        time.sleep(max(0, 1.0 / 60 - (current_time - previous_time)))
        # print(f"Time: {(time.time() - previous_time)}")
        previous_time = current_time

    # Stop keyboard controller
    keyboard.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Motion Matching Visualization and Generation")
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "generate"],
        help="Mode: interactive (Viser) or generate (Headless)",
    )
    parser.add_argument("--scenario", type=str, help="Name of the scenario function in motion_matching.scenarios")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing generated files")
    parser.add_argument("--jobs", type=int, default=0, help="Number of parallel jobs (default: 0 = all CPUs)")

    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
