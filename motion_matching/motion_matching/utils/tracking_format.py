"""Convert qpos motion to the FAR-tracking npz schema.

The schema (matching ``tracking_motion_animator.py``):

    fps:            (1,) int64
    joint_pos:      (N, 29) float32
    joint_vel:      (N, 29) float32
    body_pos_w:     (N, 30, 3) float32  -- world body positions
    body_quat_w:    (N, 30, 4) float32  -- world body quats (wxyz)
    body_lin_vel_w: (N, 30, 3) float32
    body_ang_vel_w: (N, 30, 3) float32

Ported from ``tracking_utils/interaction_mesh_to_sim_data.py`` in the sibling
``motion_matching`` repo. Joint/body remapping constants are reproduced
verbatim so the output stays bit-compatible with the existing conversion
pipeline.
"""

from __future__ import annotations

import numpy as np
import mujoco as mj

from motion_matching.utils import G1_MJCF


# Reorderings matching tracking_utils/interaction_mesh_to_sim_data.py:190-198.
_JOINT_MAPPING = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
     18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
_BODY_MAPPING = np.array(
    [0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 23, 5, 11, 17, 24, 6, 12, 18,
     25, 19, 26, 20, 27, 21, 28, 22, 29],
    dtype=np.int64,
)
_EXCLUDED_BODIES = ("world", "left_toe_link", "right_toe_link")
_EXPECTED_BODIES = 30
_EXPECTED_JOINTS = 29

_model: mj.MjModel | None = None
_body_filter: np.ndarray | None = None


def _get_model() -> tuple[mj.MjModel, np.ndarray]:
    """Load the G1 MJCF once per process and cache it (and its body filter)."""
    global _model, _body_filter
    if _model is None:
        _model = mj.MjModel.from_xml_path(G1_MJCF)
        body_ids = [
            i for i in range(_model.nbody)
            if mj.mj_id2name(_model, mj.mjtObj.mjOBJ_BODY, i) not in _EXCLUDED_BODIES
        ]
        if len(body_ids) != _EXPECTED_BODIES:
            raise RuntimeError(
                f"G1 MJCF produced {len(body_ids)} bodies after filtering, "
                f"expected {_EXPECTED_BODIES}"
            )
        _body_filter = np.array(body_ids, dtype=np.int64)
    return _model, _body_filter


def _angular_velocity(q1: np.ndarray, q2: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity from a pair of (wxyz) quaternions over ``dt``."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    w1, x1, y1, z1 = q2
    w2, x2, y2, z2 = q1_conj
    dq = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    if abs(dq[0]) < 1e-6:
        return np.zeros(3)
    return (2.0 / dt) * dq[1:] / dq[0]


def qpos_to_tracking(qpos: np.ndarray, fps: int) -> dict[str, np.ndarray]:
    """Convert ``(N, 36)`` qpos to the FAR-tracking schema.

    ``qpos`` layout is ``[root_quat_wxyz, root_pos, dof29]`` (the
    motion-matching convention). MuJoCo uses ``[root_pos, root_quat_wxyz,
    dof29]``, so the layout is swapped once before calling FK.
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 7 + _EXPECTED_JOINTS:
        raise ValueError(
            f"qpos must be (N, {7 + _EXPECTED_JOINTS}), got {qpos.shape}"
        )

    model, body_filter = _get_model()
    data = mj.MjData(model)
    num_frames = qpos.shape[0]
    num_bodies = body_filter.shape[0]

    # [quat, pos, dof] -> MuJoCo's [pos, quat, dof].
    mj_qpos = np.concatenate([qpos[:, 4:7], qpos[:, :4], qpos[:, 7:]], axis=1)

    joint_pos = qpos[:, 7:].copy()
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)

    for i in range(num_frames):
        data.qpos[:] = 0
        data.qpos[: mj_qpos.shape[1]] = mj_qpos[i]
        mj.mj_forward(model, data)
        body_pos_w[i] = data.xpos[body_filter]
        body_quat_w[i] = data.xquat[body_filter]

    dt = 1.0 / fps
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)

    body_ang_vel_w = np.zeros_like(body_lin_vel_w)
    for i in range(num_frames):
        if i == 0:
            a, b, step = body_quat_w[0], body_quat_w[1], dt
        elif i == num_frames - 1:
            a, b, step = body_quat_w[i - 1], body_quat_w[i], dt
        else:
            a, b, step = body_quat_w[i - 1], body_quat_w[i + 1], 2.0 * dt
        for j in range(num_bodies):
            body_ang_vel_w[i, j] = _angular_velocity(a[j], b[j], step)

    return {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": joint_pos[:, _JOINT_MAPPING],
        "joint_vel": joint_vel[:, _JOINT_MAPPING],
        "body_pos_w": body_pos_w[:, _BODY_MAPPING],
        "body_quat_w": body_quat_w[:, _BODY_MAPPING],
        "body_lin_vel_w": body_lin_vel_w[:, _BODY_MAPPING],
        "body_ang_vel_w": body_ang_vel_w[:, _BODY_MAPPING],
    }
