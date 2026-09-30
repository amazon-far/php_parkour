import numpy as np


def quat_mul_vec3(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    return np.array(
        [
            vx * (1 - 2 * y * y - 2 * z * z) + vy * (2 * x * y - 2 * w * z) + vz * (2 * x * z + 2 * w * y),
            vx * (2 * x * y + 2 * w * z) + vy * (1 - 2 * x * x - 2 * z * z) + vz * (2 * y * z - 2 * w * x),
            vx * (2 * x * z - 2 * w * y) + vy * (2 * y * z + 2 * w * x) + vz * (1 - 2 * x * x - 2 * y * y),
        ]
    )


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    norm = w * w + x * x + y * y + z * z
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([w, -x, -y, -z], dtype=np.float32) / norm


def quat_inv_mul_vec3(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul_vec3(quat_inv(q), v)


def quat_mul_inv(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return quat_mul(q1, quat_inv(q2))


def quat_inv_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Inverse multiply: q1^(-1) * q2"""
    return quat_mul(quat_inv(q1), q2)


def quat_abs(q: np.ndarray) -> np.ndarray:
    """Get absolute value of quaternion (ensure w >= 0)"""
    return q if q[0] >= 0.0 else -q


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion"""
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


def quat_to_scaled_angle_axis(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to scaled angle-axis representation"""
    # Extract the axis and angle
    angle = 2.0 * np.arccos(np.clip(abs(q[..., 0]), 0, 1))

    # Handle both batch and single quaternion cases
    if q.ndim == 1:
        # Single quaternion
        if angle > 1e-8:
            axis = q[1:4] / np.sin(angle * 0.5)
            return axis * angle
        else:
            return np.zeros(3, dtype=np.float32)
    else:
        # Batch of quaternions
        result = np.zeros((q.shape[0], 3), dtype=np.float32)
        # Where angle is significant, compute the scaled axis
        valid_mask = angle > 1e-8
        if np.any(valid_mask):
            axis = q[valid_mask, 1:4] / np.sin(angle[valid_mask] * 0.5)[..., np.newaxis]
            result[valid_mask] = axis * angle[valid_mask, np.newaxis]
        return result


def quat_from_scaled_angle_axis(saa: np.ndarray) -> np.ndarray:
    """Convert scaled angle-axis to quaternion"""
    angle = np.linalg.norm(saa)
    if angle > 1e-8:
        axis = saa / angle
        half_angle = angle * 0.5
        s = np.sin(half_angle)
        return np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)
    else:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    """Create quaternion from angle and axis"""
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max - matches C++ implementation"""
    return max(min_val, min(value, max_val))


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector"""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def length(v: np.ndarray) -> float:
    """Vector length"""
    return float(np.linalg.norm(v))


# C++ style trajectory prediction functions
def halflife_to_damping(halflife: float, eps: float = 1e-5) -> float:
    """Convert halflife to damping coefficient"""
    LN2 = 0.693147180559945309417
    return (4.0 * LN2) / (halflife + eps)


def fast_negexpf(x: float) -> float:
    """Fast negative exponential approximation"""
    return 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x)
