"""Kinematics utilities for coordinate transformations and body orientation."""

import numpy as np
import mujoco


def get_body_xmat(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Get the 3x3 rotation matrix of a body."""
    return data.xmat[body_id].reshape(3, 3)


def get_forward_vector(xmat: np.ndarray) -> np.ndarray:
    """Get the forward direction vector (x-axis) from a rotation matrix."""
    forward = xmat[:, 0]
    return forward / (np.linalg.norm(forward) + 1e-8)


def get_up_vector(xmat: np.ndarray) -> np.ndarray:
    """Get the up direction vector (z-axis) from a rotation matrix."""
    up = xmat[:, 2]
    return up / (np.linalg.norm(up) + 1e-8)


def get_right_vector(xmat: np.ndarray) -> np.ndarray:
    """Get the right direction vector (y-axis) from a rotation matrix."""
    right = xmat[:, 1]
    return right / (np.linalg.norm(right) + 1e-8)


def rotate_to_local_frame(vec: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into a local body frame."""
    return xmat.T @ vec


def rotate_to_world_frame(vec: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    """Rotate a local body frame vector into world frame."""
    return xmat @ vec


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def compute_facing_alignment(fwd1: np.ndarray, fwd2: np.ndarray) -> float:
    """Compute facing alignment (dot(fwd1, -fwd2)). Higher = more face-to-face."""
    return float(np.dot(fwd1, -fwd2))


def compute_up_alignment(up1: np.ndarray, up2: np.ndarray) -> float:
    """Compute up vector alignment between two agents."""
    return float(np.dot(up1, up2))


def compute_tilt_angle(up_vec: np.ndarray) -> float:
    """Compute the tilt angle of a body from vertical (radians)."""
    world_up = np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(np.dot(up_vec, world_up), -1, 1)
    return np.arccos(cos_angle)


def compute_relative_velocity(vel1: np.ndarray, vel2: np.ndarray) -> np.ndarray:
    """Compute relative velocity between two agents."""
    return vel1 - vel2


def get_root_quat(data: mujoco.MjData, qpos_idx: np.ndarray) -> np.ndarray:
    """Extract root quaternion [w, x, y, z] from qpos."""
    return data.qpos[qpos_idx[3:7]].copy()


def get_root_position(data: mujoco.MjData, qpos_idx: np.ndarray) -> np.ndarray:
    """Extract root position [x, y, z] from qpos."""
    return data.qpos[qpos_idx[0:3]].copy()


def get_root_linear_velocity(data: mujoco.MjData, qvel_idx: np.ndarray) -> np.ndarray:
    """Extract root linear velocity [vx, vy, vz] from qvel."""
    return data.qvel[qvel_idx[0:3]].copy()


def get_root_angular_velocity(data: mujoco.MjData, qvel_idx: np.ndarray) -> np.ndarray:
    """Extract root angular velocity [wx, wy, wz] from qvel."""
    return data.qvel[qvel_idx[3:6]].copy()
