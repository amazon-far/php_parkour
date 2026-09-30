#!/usr/bin/env python3
"""URDF Kinematics Calculator

A class for loading URDF files and computing global joint positions and orientations
from root pose and joint angles (DOF).

Usage:
    from urdf_kinematics import URDFKinematics

    # Load URDF
    kin = URDFKinematics("path/to/robot.urdf")

    # Compute global joint poses
    root_pos = np.array([0.0, 0.0, 0.0])  # x, y, z
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    dof = np.array([0.1, 0.2, 0.3, ...])  # joint angles

    global_positions, global_orientations = kin.compute_global_poses(root_pos, root_quat, dof)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple
from scipy.spatial.transform import Rotation

import yourdfpy


class URDFKinematics:
    """URDF-based kinematics calculator for computing global joint positions and orientations.

    This class loads a URDF file and provides methods to compute global joint poses
    from root pose and joint angles (DOF).
    """

    def __init__(
        self,
        urdf_path: str | Path,
        load_meshes: bool = False,
        load_collision_meshes: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize URDF kinematics calculator.

        Args:
            urdf_path: Path to URDF file
            load_meshes: Whether to load visual meshes (not needed for kinematics)
            load_collision_meshes: Whether to load collision meshes (not needed for kinematics)
            verbose: If True, print what was loaded (default: quiet, suitable for a library).
        """
        self.urdf = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=load_meshes,
            build_scene_graph=True,  # Required for kinematics
            load_collision_meshes=load_collision_meshes,
            build_collision_scene_graph=load_collision_meshes,
        )
        if verbose:
            print(f"Loaded URDF from {urdf_path}")

        # Get joint information
        self.joint_names = list(self.urdf.actuated_joint_names)
        self.num_joints = len(self.joint_names)
        self.joint_limits = self._get_joint_limits()

        if verbose:
            print(f"Robot has {self.num_joints} actuated joints: {self.joint_names}")

        # Get all link names (including base link)
        self.link_names = list(self.urdf.link_map.keys())
        self.num_links = len(self.link_names)

        if verbose:
            print(f"Robot has {self.num_links} links: {self.link_names}")

    def compute_global_poses(
        self,
        root_position: np.ndarray,
        root_orientation: np.ndarray,
        joint_angles: np.ndarray,
        return_link_names: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, list[str]]:
        """Compute global positions and orientations for all joints/links.

        Args:
            root_position: Root position [x, y, z] in meters
            root_orientation: Root orientation as quaternion [w, x, y, z]
            joint_angles: Joint angles in radians, shape (num_joints,)
            return_link_names: Whether to return link names along with poses

        Returns:
            global_positions: Array of shape (num_links, 3) with global positions
            global_orientations: Array of shape (num_links, 4) with quaternions [w, x, y, z]
            link_names: List of link names (if return_link_names=True)
        """
        # Validate inputs
        if len(root_position) != 3:
            raise ValueError(f"root_position must have 3 elements, got {len(root_position)}")

        if len(root_orientation) != 4:
            raise ValueError(f"root_orientation must have 4 elements, got {len(root_orientation)}")

        if len(joint_angles) != self.num_joints:
            raise ValueError(f"joint_angles must have {self.num_joints} elements, got {len(joint_angles)}")

        # Normalize quaternion
        root_quat = self._normalize_quaternion(root_orientation)

        # Update URDF configuration
        self.urdf.update_cfg(joint_angles)

        # Initialize output arrays
        global_positions = np.zeros((self.num_links, 3))
        global_orientations = np.zeros((self.num_links, 4))

        # Compute poses for each link
        for i, link_name in enumerate(self.link_names):
            # Get transform from world to this link
            T_world_link = self.urdf.get_transform(link_name, self.urdf.base_link)

            transformed_position, transformed_quat_wxyz = self._transform_link_pose(
                T_world_link, root_quat, root_position
            )

            global_positions[i] = transformed_position
            global_orientations[i] = transformed_quat_wxyz

        if return_link_names:
            return global_positions, global_orientations, self.link_names
        else:
            return global_positions, global_orientations

    def compute_joint_poses(
        self,
        root_position: np.ndarray,
        root_orientation: np.ndarray,
        joint_angles: np.ndarray,
        return_joint_names: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, list[str]]:
        """Compute global positions and orientations for actuated joints only.

        Args:
            root_position: Root position [x, y, z] in meters
            root_orientation: Root orientation as quaternion [w, x, y, z]
            joint_angles: Joint angles in radians, shape (num_joints,)
            return_joint_names: Whether to return joint names along with poses

        Returns:
            global_positions: Array of shape (num_joints, 3) with global positions
            global_orientations: Array of shape (num_joints, 4) with quaternions [w, x, y, z]
            joint_names: List of joint names (if return_joint_names=True)
        """
        # Validate inputs
        if len(root_position) != 3:
            raise ValueError(f"root_position must have 3 elements, got {len(root_position)}")

        if len(root_orientation) != 4:
            raise ValueError(f"root_orientation must have 4 elements, got {len(root_orientation)}")

        if len(joint_angles) != self.num_joints:
            raise ValueError(f"joint_angles must have {self.num_joints} elements, got {len(joint_angles)}")

        # Normalize quaternion
        root_quat = self._normalize_quaternion(root_orientation)

        # Update URDF configuration
        self.urdf.update_cfg(joint_angles)

        # Initialize output arrays
        global_positions = np.zeros((self.num_joints, 3))
        global_orientations = np.zeros((self.num_joints, 4))

        # Compute poses for each actuated joint
        for i, joint_name in enumerate(self.joint_names):
            joint = self.urdf.joint_map[joint_name]
            child_link = joint.child

            # Get transform from world to child link
            T_world_link = self.urdf.get_transform(child_link, self.urdf.base_link)

            transformed_position, transformed_quat_wxyz = self._transform_link_pose(
                T_world_link, root_quat, root_position
            )

            global_positions[i] = transformed_position
            global_orientations[i] = transformed_quat_wxyz

        if return_joint_names:
            return global_positions, global_orientations, self.joint_names
        else:
            return global_positions, global_orientations

    def get_joint_limits(self) -> dict[str, tuple[float | None, float | None]]:
        """Get joint limits for all actuated joints.

        Returns:
            Dictionary mapping joint names to (lower_limit, upper_limit) tuples.
            None values indicate no limit.
        """
        return self.joint_limits.copy()

    def get_joint_names(self) -> list[str]:
        """Get names of all actuated joints.

        Returns:
            List of joint names in order.
        """
        return self.joint_names.copy()

    def get_link_names(self) -> list[str]:
        """Get names of all links.

        Returns:
            List of link names in order.
        """
        return self.link_names.copy()

    def _get_joint_limits(self) -> dict[str, tuple[float | None, float | None]]:
        """Get joint limits for all actuated joints.

        Returns:
            Dictionary mapping joint names to (lower_limit, upper_limit) tuples.
            None values indicate no limit.
        """
        limits = {}
        for joint_name, joint in zip(self.urdf.actuated_joint_names, self.urdf.actuated_joints):
            if joint.limit is None:
                limits[joint_name] = (-np.pi, np.pi)
            else:
                limits[joint_name] = (joint.limit.lower, joint.limit.upper)
        return limits

    def _normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """Normalize quaternion and ensure it's in [w, x, y, z] format.

        Args:
            quat: Quaternion as [w, x, y, z]

        Returns:
            Normalized quaternion as [w, x, y, z]
        """
        quat = np.array(quat)

        if len(quat) != 4:
            raise ValueError(f"Quaternion must have 4 elements, got {len(quat)}")

        # Assume [w, x, y, z] format
        w, x, y, z = quat

        # Normalize
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        return np.array([w / norm, x / norm, y / norm, z / norm])

    def _transform_link_pose(
        self, T_world_link: np.ndarray, root_quat: np.ndarray, root_position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a root transform to one link's world pose.

        Shared inner step of ``compute_global_poses`` and ``compute_joint_poses``.

        Args:
            T_world_link: 4x4 transform of the link relative to the base link.
            root_quat: Normalized root orientation as ``[w, x, y, z]``.
            root_position: Root position ``[x, y, z]``.

        Returns:
            ``(transformed_position, transformed_quat_wxyz)`` — the link's pose in
            the root frame, with the quaternion as ``[w, x, y, z]``.
        """
        # Extract position and orientation
        position = T_world_link[:3, 3]
        rotation_matrix = T_world_link[:3, :3]

        # Convert rotation matrix to quaternion
        r = Rotation.from_matrix(rotation_matrix)
        quat = r.as_quat()  # Returns [x, y, z, w]

        # Apply root transform
        root_rotation = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        # np.array() forces a writable copy: ``position`` is a read-only view
        # into the transform matrix, and scipy's Rotation.apply rejects
        # read-only buffers (newer scipy).
        transformed_position = root_rotation.apply(np.array(position)) + root_position
        transformed_quat = root_rotation * Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
        transformed_quat_xyzw = transformed_quat.as_quat()
        transformed_quat_wxyz = np.array(
            [transformed_quat_xyzw[3], transformed_quat_xyzw[0], transformed_quat_xyzw[1], transformed_quat_xyzw[2]]
        )

        return transformed_position, transformed_quat_wxyz
