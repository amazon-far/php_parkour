from motion_matching.kinematics import quat as quat
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage
import struct
import numpy as np
import argparse
import json
import os
from pathlib import Path

from motion_matching.kinematics.urdf_kinematics import URDFKinematics
from motion_matching.utils import G1_URDF, db_path, ROBOT_MOTIONS_DIR
from motion_matching.utils.data_utils import animation_mirror_xz, mirror_terrain


def main() -> None:
    """ Parse Command Line Arguments """

    parser = argparse.ArgumentParser(description="Generate motion database for G1 robot")
    parser.add_argument(
        "--output",
        "-o",
        default="database_g1_climb_76_high_speed_up_test.bin",
        help=(
            "Output database filename. Must be a bare filename (no slashes, no '..', "
            "not absolute); resolved against PHP_MOTION_DATABASE_DIR."
        ),
    )
    parser.add_argument(
        "--motion-file", type=Path,
        default=ROBOT_MOTIONS_DIR / "climb_76_high_speed_up.npz",
        help="Source G1 NPZ containing qpos: [root_quat_wxyz, root_pos, dof29].",
    )
    parser.add_argument(
        "--terrain-file", type=Path,
        default=Path(db_path("terrain/climb_76_high_speed/box.obj")),
        help="Terrain OBJ paired with the source motion; defaults to the released sample terrain.",
    )
    parser.add_argument("--fps", type=float, default=30, help="Source motion frame rate.")
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")

    # Reject anything other than a bare filename.
    _out_arg = Path(args.output)
    if _out_arg.is_absolute() or len(_out_arg.parts) != 1 or _out_arg.name in ("", ".", ".."):
        raise ValueError(f"--output must be a bare filename (no path separators), got: {args.output!r}")
    args.output = db_path(args.output)

    # NOTE: Edit this list to point at the motion(s) you want to bake into the
    # database. Each entry is [npz_file, add_mirror, speed_up, fps, terrain_file].
    # Remember to update --output (or its default above) to a matching filename.
    files = [
        [
            str(args.motion_file),  # g1_skill_motion
            True,  # add_mirror
            1,  # speed_up (1 is original speed)
            args.fps,  # motion fps
            str(args.terrain_file),  # terrain_file
        ]
    ]

    # Validate that every input file exists before any heavy work.
    for entry in files:
        npz_file, _, _, _, terrain_file = entry
        if not Path(npz_file).is_file():
            raise FileNotFoundError(f"Retargeted motion file not found: {npz_file}")
        if not Path(terrain_file).is_file():
            raise FileNotFoundError(f"Terrain file not found: {terrain_file}")

    # Ensure output directory exists.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if len(files[0]) == 5:
        terrain_files = list(set(terrain_file for _, _, _, _, terrain_file in files))

    """ Basic function for mirroring animation data with this particular skeleton structure """

    """ Process URDF Kinematics """

    kin = URDFKinematics(urdf_path=G1_URDF, load_meshes=False, load_collision_meshes=False)

    # Build parent array
    link_to_joint_idx = {kin.urdf.joint_map[name].child: i for i, name in enumerate(kin.joint_names)}

    joint_parents = np.full(len(kin.joint_names), -1, dtype=np.int32)

    for i, joint_name in enumerate(kin.joint_names):
        joint = kin.urdf.joint_map[joint_name]
        parent_link = joint.parent

        if parent_link in link_to_joint_idx:
            joint_parents[i] = link_to_joint_idx[parent_link]
        elif parent_link == kin.urdf.base_link:
            joint_parents[i] = -1

    global_positions, global_orientations = kin.compute_joint_poses(
        root_position=np.array([0.0, 0.0, 0.0]),
        root_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        joint_angles=np.zeros(kin.num_joints),
    )

    # Initialize array to store bone vectors (num_joints x 3)
    joint_bone_vectors = np.zeros((kin.num_joints, 3))

    print("Computing bone vectors for each joint...")
    for i, joint_name in enumerate(kin.joint_names):
        parent_idx = joint_parents[i]

        # Find indices for parent and child links
        if parent_idx != -1:
            parent_pos = global_positions[parent_idx]
            child_pos = global_positions[i]

            # Bone vector = child position - parent position
            joint_bone_vectors[i] = child_pos - parent_pos
            print(
                f"{i:2d}: {joint_name:30s} -> [{joint_bone_vectors[i][0]:6.3f}, {joint_bone_vectors[i][1]:6.3f}, {joint_bone_vectors[i][2]:6.3f}]"
            )
        else:
            child_pos = global_positions[i]

            # Bone vector = child position - 0
            joint_bone_vectors[i] = child_pos
            print(
                f"{i:2d}: {joint_name:30s} -> [{joint_bone_vectors[i][0]:6.3f}, {joint_bone_vectors[i][1]:6.3f}, {joint_bone_vectors[i][2]:6.3f}]"
            )

    print(f"\nJoint bone vectors shape: {joint_bone_vectors.shape}")

    # Add root bone vector
    full_bone_vectors = np.concatenate([np.zeros((1, 3)), joint_bone_vectors], axis=0)
    print(f"Full bone vectors shape: {full_bone_vectors.shape}")

    # Get rotation axis for each joint (DOF)
    print("\nComputing joint rotation axes...")
    joint_axes = np.zeros((kin.num_joints, 3))
    for i, joint_name in enumerate(kin.joint_names):
        joint = kin.urdf.joint_map[joint_name]
        if hasattr(joint, "axis") and joint.axis is not None:
            joint_axes[i] = np.array(joint.axis)
            print(
                f"{i:2d}: {joint_name:30s} -> axis: [{joint_axes[i][0]:6.3f}, {joint_axes[i][1]:6.3f}, {joint_axes[i][2]:6.3f}]"
            )

    # Add root axis (zero for root joint)
    print(f"Joint axes shape: {joint_axes.shape}")

    # Add root parent
    joint_parents_w_root = np.concatenate([[-1], joint_parents + 1])

    # Build name array
    joint_names_w_root = [kin.urdf.base_link] + kin.joint_names

    """ We will accumulate data in these lists """

    bone_positions = []
    bone_velocities = []
    bone_rotations = []
    bone_angular_velocities = []
    bone_parents = []
    bone_names = []

    range_starts = []
    range_stops = []

    contact_states = []

    """ Loop Over Files """

    for filename, add_mirror, speed_up, fps, _ in files:
        # For each file we process it mirrored and not mirrored
        for mirror in [False, True]:
            if add_mirror is False and mirror is True:
                continue

            """ Load Data """

            print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))

            data = np.load(filename)
            num_frames = data["qpos"].shape[0]
            dof = data["qpos"][:, 7:]
            root_wxyz = data["qpos"][:, :4]
            root_pos = data["qpos"][:, 4:7]

            assert dof.shape[1] == joint_axes.shape[0]

            positions = np.repeat(full_bone_vectors[np.newaxis, :, :], num_frames, axis=0)
            positions[:, 0] = root_pos  # T x (J+1) x 3
            rotations = quat.unroll(quat.from_angle_axis(dof, joint_axes))
            rotations = np.concatenate([root_wxyz[:, np.newaxis, :], rotations], axis=1)
            rotations = quat.unroll(rotations)  # T x (J+1) x 4

            # Convert from cm to m
            # positions *= 0.01

            if mirror:
                dof, root_pos, root_wxyz = animation_mirror_xz(
                    dof=dof, root_pos=root_pos, root_wxyz=root_wxyz, names=kin.joint_names
                )
                positions[:, 0] = root_pos  # T x (J+1) x 3
                rotations = quat.unroll(quat.from_angle_axis(dof, joint_axes))
                rotations = np.concatenate([root_wxyz[:, np.newaxis, :], rotations], axis=1)
                rotations = quat.unroll(rotations)  # T x (J+1) x 4

            """ Supersample """
            frame_ratio = 60.0 / fps / speed_up
            if not np.abs(frame_ratio - 1.0) < 1e-2:
                nframes = positions.shape[0]
                nbones = positions.shape[1]

                # Supersample data to 60 fps
                original_times = np.linspace(0, nframes - 1, nframes)
                sample_times = np.linspace(0, nframes - 1, int(frame_ratio * nframes))

                # This does a cubic interpolation of the data for supersampling
                positions = griddata(
                    original_times, positions.reshape([nframes, -1]), sample_times, method="cubic"
                ).reshape([len(sample_times), nbones, 3])
                rotations = griddata(
                    original_times, rotations.reshape([nframes, -1]), sample_times, method="cubic"
                ).reshape([len(sample_times), nbones, 4])

                # Need to re-normalize after super-sampling
                rotations = quat.normalize(rotations)

            """ Extract Simulation Bone """

            # First compute world space positions/rotations
            global_rotations, global_positions = quat.fk(lrot=rotations, lpos=positions, parents=joint_parents_w_root)

            # Specify joints to use for simulation bone
            sim_position_joints = np.array(
                [
                    joint_names_w_root.index("left_shoulder_pitch_joint"),
                    joint_names_w_root.index("right_shoulder_pitch_joint"),
                ]
            )
            sim_rotation_joint = joint_names_w_root.index(kin.urdf.base_link)

            # Position comes from spine joint
            sim_position = np.array([1.0, 1.0, 0.0]) * global_positions[:, sim_position_joints].mean(
                axis=1, keepdims=True
            )
            sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode="interp")

            # Direction comes from projected hip forward direction
            sim_direction = np.array([1.0, 1.0, 0.0]) * quat.mul_vec(
                global_rotations[:, sim_rotation_joint : sim_rotation_joint + 1], np.array([1.0, 0.0, 0.0])
            )

            # We need to re-normalize the direction after both projection and smoothing
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[..., np.newaxis]
            if sim_direction.shape[0] > 61:
                sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode="interp")
            else:
                sim_direction = signal.savgol_filter(sim_direction, sim_direction.shape[0], 3, axis=0, mode="interp")
            sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[..., np.newaxis])

            # Extract rotation from direction
            sim_rotation = quat.normalize(quat.between(np.array([1, 0, 0]), sim_direction))

            # Transform first joints to be local to sim and append sim as root bone
            positions[:, 0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:, 0:1] - sim_position)
            rotations[:, 0:1] = quat.mul(quat.inv(sim_rotation), rotations[:, 0:1])

            positions = np.concatenate([sim_position, positions], axis=1)
            rotations = np.concatenate([sim_rotation, rotations], axis=1)

            bone_parents = np.concatenate([[-1], joint_parents_w_root + 1])

            bone_names = ["Simulation"] + joint_names_w_root

            """ Compute Velocities """

            # Compute velocities via central difference
            velocities = np.empty_like(positions)
            velocities[1:-1] = (
                0.5 * (positions[2:] - positions[1:-1]) * 60.0 + 0.5 * (positions[1:-1] - positions[:-2]) * 60.0
            )
            velocities[0] = velocities[1] - (velocities[3] - velocities[2])
            velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

            # Same for angular velocities
            angular_velocities = np.zeros_like(positions)
            angular_velocities[1:-1] = (
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) * 60.0
                + 0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2]))) * 60.0
            )
            angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
            angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

            """ Compute Contact Data """

            global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
                lrot=rotations, lpos=positions, lvel=velocities, lang=angular_velocities, parents=bone_parents
            )

            contact_velocity_threshold = 0.5

            contact_velocity = np.sqrt(
                np.sum(
                    global_velocities[
                        :,
                        np.array(
                            [bone_names.index("left_ankle_roll_joint"), bone_names.index("right_ankle_roll_joint")]
                        ),
                    ]
                    ** 2,
                    axis=-1,
                )
            )

            # Contacts are given for when contact bones are below velocity threshold
            contacts = contact_velocity < contact_velocity_threshold

            # Median filter here acts as a kind of "majority vote", and removes
            # small regions  where contact is either active or inactive
            for ci in range(contacts.shape[1]):
                contacts[:, ci] = ndimage.median_filter(contacts[:, ci], size=6, mode="nearest")

            """ Append to Database """

            bone_positions.append(positions)
            bone_velocities.append(velocities)
            bone_rotations.append(rotations)
            bone_angular_velocities.append(angular_velocities)

            offset = 0 if len(range_starts) == 0 else range_stops[-1]

            range_starts.append(offset)
            range_stops.append(offset + len(positions))

            contact_states.append(contacts)

    """ Concatenate Data """

    bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
    bone_velocities = np.concatenate(bone_velocities, axis=0).astype(np.float32)
    bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
    bone_angular_velocities = np.concatenate(bone_angular_velocities, axis=0).astype(np.float32)
    bone_parents = bone_parents.astype(np.int32)

    range_starts = np.array(range_starts).astype(np.int32)
    range_stops = np.array(range_stops).astype(np.int32)

    contact_states = np.concatenate(contact_states, axis=0).astype(np.uint8)

    """ Write Database """

    print("Writing Database...")

    with open(args.output, "wb") as f:
        nframes = bone_positions.shape[0]
        nbones = bone_positions.shape[1]
        nranges = range_starts.shape[0]
        ncontacts = contact_states.shape[1]

        f.write(struct.pack("II", nframes, nbones) + bone_positions.ravel().tobytes())
        f.write(struct.pack("II", nframes, nbones) + bone_velocities.ravel().tobytes())
        f.write(struct.pack("II", nframes, nbones) + bone_rotations.ravel().tobytes())
        f.write(struct.pack("II", nframes, nbones) + bone_angular_velocities.ravel().tobytes())
        f.write(struct.pack("I", nbones) + bone_parents.ravel().tobytes())

        f.write(struct.pack("I", nranges) + range_starts.ravel().tobytes())
        f.write(struct.pack("I", nranges) + range_stops.ravel().tobytes())

        f.write(struct.pack("II", nframes, ncontacts) + contact_states.ravel().tobytes())

    """ Process Terrain Files """
    if terrain_files is not None:
        print("\nProcessing terrain files...")
        for terrain_file in terrain_files:
            mirror_terrain(terrain_file)

            # Mirror corresponding JSON file if it exists
            json_file = os.path.splitext(terrain_file)[0] + ".json"
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    json_data = json.load(f)

                # Mirror the JSON data
                # Format: pos=[x, y, z], quat=[w, x, y, z], size=[x, y, z]
                # Mirror Y: pos_y -> -pos_y (index 1)
                json_data["pos"][1] *= -1
                # Mirror Rotation: quat_x -> -quat_x (index 1), quat_z -> -quat_z (index 3)
                json_data["quat"][1] *= -1
                json_data["quat"][3] *= -1

                # Save mirrored JSON file
                base, ext = os.path.splitext(json_file)
                json_output_file = base + "_mirror" + ext
                with open(json_output_file, "w") as f:
                    json.dump(json_data, f, indent=4)

                print(f"Mirrored JSON: {json_file} -> {json_output_file}")


if __name__ == "__main__":
    main()
