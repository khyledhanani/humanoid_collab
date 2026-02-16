"""Retargeting utilities for AMASS (SMPL) to MuJoCo humanoid skeleton.

SMPL skeleton has 24 joints in the following order:
0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee,
6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3, 10: L_Foot, 11: R_Foot,
12: Neck, 13: L_Collar, 14: R_Collar, 15: Head, 16: L_Shoulder, 17: R_Shoulder,
18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist, 22: L_Hand, 23: R_Hand

MuJoCo humanoid joint order (for one agent, qpos after root 7 DoF):
0: head_yaw, 1: head_pitch
2: left_shoulder1, 3: left_shoulder2, 4: left_elbow
5: right_shoulder1, 6: right_shoulder2, 7: right_elbow
8: abdomen_z, 9: abdomen_y
10: left_hip_x, 11: left_hip_z, 12: left_hip_y
13: left_knee
14: left_ankle_roll, 15: left_ankle, 16: left_toe
17: right_hip_x, 18: right_hip_z, 19: right_hip_y
20: right_knee
21: right_ankle_roll, 22: right_ankle, 23: right_toe
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None

from humanoid_collab.amp.motion_data import MotionClip


@dataclass
class SkeletonConfig:
    """Configuration for the target MuJoCo skeleton.

    Stores joint names, limits, and ordering for retargeting.
    """
    # Number of qpos/qvel entries per agent (including root)
    nq: int = 31  # 7 root + 24 joints
    nv: int = 30  # 6 root + 24 DoF

    # Joint names in qpos order (after root)
    joint_names: Tuple[str, ...] = (
        "head_yaw", "head_pitch",
        "left_shoulder1", "left_shoulder2",
        "left_elbow",
        "right_shoulder1", "right_shoulder2",
        "right_elbow",
        "abdomen_z", "abdomen_y",
        "left_hip_x", "left_hip_z", "left_hip_y",
        "left_knee",
        "left_ankle_roll",
        "left_ankle",
        "left_toe",
        "right_hip_x", "right_hip_z", "right_hip_y",
        "right_knee",
        "right_ankle_roll",
        "right_ankle",
        "right_toe",
    )

    # Joint limits (degrees) - matches MJCF
    joint_limits_deg: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        if self.joint_limits_deg is None:
            self.joint_limits_deg = {
                "abdomen_z": (-45, 45),
                "abdomen_y": (-75, 30),
                "left_hip_x": (-25, 5),
                "left_hip_z": (-60, 35),
                "left_hip_y": (-110, 20),
                "left_knee": (0, 160),
                "left_ankle_roll": (-25, 25),
                "left_ankle": (-50, 50),
                "left_toe": (-35, 55),
                "right_hip_x": (-25, 5),
                "right_hip_z": (-60, 35),
                "right_hip_y": (-110, 20),
                "right_knee": (0, 160),
                "right_ankle_roll": (-25, 25),
                "right_ankle": (-50, 50),
                "right_toe": (-35, 55),
                "left_shoulder1": (-85, 60),
                "left_shoulder2": (-85, 60),
                "left_elbow": (-90, 50),
                "right_shoulder1": (-60, 85),
                "right_shoulder2": (-60, 85),
                "right_elbow": (-90, 50),
                "head_yaw": (-50, 50),
                "head_pitch": (-30, 30),
            }


# SMPL joint indices
SMPL_JOINTS = {
    "pelvis": 0,
    "l_hip": 1,
    "r_hip": 2,
    "spine1": 3,
    "l_knee": 4,
    "r_knee": 5,
    "spine2": 6,
    "l_ankle": 7,
    "r_ankle": 8,
    "spine3": 9,
    "l_foot": 10,
    "r_foot": 11,
    "neck": 12,
    "l_collar": 13,
    "r_collar": 14,
    "head": 15,
    "l_shoulder": 16,
    "r_shoulder": 17,
    "l_elbow": 18,
    "r_elbow": 19,
    "l_wrist": 20,
    "r_wrist": 21,
    "l_hand": 22,
    "r_hand": 23,
}


def axis_angle_to_euler(axis_angle: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Convert axis-angle representation to Euler angles.

    Args:
        axis_angle: Axis-angle rotation vector (3,)
        seq: Euler angle sequence

    Returns:
        Euler angles in radians
    """
    if Rotation is None:
        raise ImportError("scipy is required for retargeting")

    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.zeros(3)

    rot = Rotation.from_rotvec(axis_angle)
    return rot.as_euler(seq, degrees=False)


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to quaternion (w, x, y, z).

    Args:
        axis_angle: Axis-angle rotation vector (3,)

    Returns:
        Quaternion as (w, x, y, z)
    """
    if Rotation is None:
        raise ImportError("scipy is required for retargeting")

    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])

    rot = Rotation.from_rotvec(axis_angle)
    quat_xyzw = rot.as_quat()  # scipy returns (x, y, z, w)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def extract_joint_angle(
    pose: np.ndarray,
    joint_idx: int,
    axis: str,
) -> float:
    """Extract a single joint angle from SMPL pose.

    Args:
        pose: SMPL pose array (72,) = 24 joints * 3 axis-angle
        joint_idx: SMPL joint index
        axis: Which Euler axis to extract ('x', 'y', or 'z')

    Returns:
        Joint angle in radians
    """
    axis_angle = pose[joint_idx * 3:(joint_idx + 1) * 3]
    euler = axis_angle_to_euler(axis_angle, seq="xyz")

    axis_map = {"x": 0, "y": 1, "z": 2}
    return euler[axis_map[axis]]


def retarget_smpl_frame(
    pose: np.ndarray,
    trans: np.ndarray,
    skeleton: SkeletonConfig,
    height_offset: float = 0.0,
) -> np.ndarray:
    """Retarget a single SMPL frame to MuJoCo humanoid qpos.

    Args:
        pose: SMPL pose parameters (72,) - 24 joints * 3 axis-angle
        trans: SMPL translation (3,)
        skeleton: Target skeleton configuration
        height_offset: Additional height offset to add

    Returns:
        MuJoCo qpos array (nq,)
    """
    qpos = np.zeros(skeleton.nq)

    # Root position (first 3 entries)
    qpos[0] = trans[0]  # x
    qpos[1] = trans[1]  # y
    qpos[2] = trans[2] + height_offset  # z (with offset for ground clearance)

    # Root orientation (quaternion, entries 3-6)
    root_axis_angle = pose[0:3]
    qpos[3:7] = axis_angle_to_quat(root_axis_angle)

    # Map SMPL joints to MuJoCo joints. We first compute joint angles by name,
    # then pack them in exact model qpos order from skeleton.joint_names.
    joint_angles: Dict[str, float] = {}

    # Abdomen (from spine1)
    spine_aa = pose[SMPL_JOINTS["spine1"] * 3:(SMPL_JOINTS["spine1"] + 1) * 3]
    spine_euler = axis_angle_to_euler(spine_aa)
    joint_angles["abdomen_z"] = np.clip(spine_euler[2], np.radians(-45), np.radians(45))
    joint_angles["abdomen_y"] = np.clip(spine_euler[1], np.radians(-75), np.radians(30))

    # Left hip (3 DoF)
    l_hip_aa = pose[SMPL_JOINTS["l_hip"] * 3:(SMPL_JOINTS["l_hip"] + 1) * 3]
    l_hip_euler = axis_angle_to_euler(l_hip_aa)
    joint_angles["left_hip_x"] = np.clip(l_hip_euler[0], np.radians(-25), np.radians(5))
    joint_angles["left_hip_z"] = np.clip(l_hip_euler[2], np.radians(-60), np.radians(35))
    joint_angles["left_hip_y"] = np.clip(l_hip_euler[1], np.radians(-110), np.radians(20))

    # Left knee
    l_knee_aa = pose[SMPL_JOINTS["l_knee"] * 3:(SMPL_JOINTS["l_knee"] + 1) * 3]
    l_knee_euler = axis_angle_to_euler(l_knee_aa)
    # MuJoCo knee flexion is positive (range [0, 160] deg). SMPL knee flexion comes out negative here.
    joint_angles["left_knee"] = np.clip(-l_knee_euler[1], np.radians(0), np.radians(160))

    # Left ankle complex
    l_ankle_aa = pose[SMPL_JOINTS["l_ankle"] * 3:(SMPL_JOINTS["l_ankle"] + 1) * 3]
    l_ankle_euler = axis_angle_to_euler(l_ankle_aa)
    joint_angles["left_ankle_roll"] = np.clip(l_ankle_euler[0], np.radians(-25), np.radians(25))
    joint_angles["left_ankle"] = np.clip(l_ankle_euler[1], np.radians(-50), np.radians(50))
    l_foot_aa = pose[SMPL_JOINTS["l_foot"] * 3:(SMPL_JOINTS["l_foot"] + 1) * 3]
    l_foot_euler = axis_angle_to_euler(l_foot_aa)
    joint_angles["left_toe"] = np.clip(l_foot_euler[1], np.radians(-35), np.radians(55))

    # Right hip (3 DoF)
    r_hip_aa = pose[SMPL_JOINTS["r_hip"] * 3:(SMPL_JOINTS["r_hip"] + 1) * 3]
    r_hip_euler = axis_angle_to_euler(r_hip_aa)
    joint_angles["right_hip_x"] = np.clip(-r_hip_euler[0], np.radians(-25), np.radians(5))
    joint_angles["right_hip_z"] = np.clip(-r_hip_euler[2], np.radians(-60), np.radians(35))
    joint_angles["right_hip_y"] = np.clip(r_hip_euler[1], np.radians(-110), np.radians(20))

    # Right knee
    r_knee_aa = pose[SMPL_JOINTS["r_knee"] * 3:(SMPL_JOINTS["r_knee"] + 1) * 3]
    r_knee_euler = axis_angle_to_euler(r_knee_aa)
    joint_angles["right_knee"] = np.clip(-r_knee_euler[1], np.radians(0), np.radians(160))

    # Right ankle complex
    r_ankle_aa = pose[SMPL_JOINTS["r_ankle"] * 3:(SMPL_JOINTS["r_ankle"] + 1) * 3]
    r_ankle_euler = axis_angle_to_euler(r_ankle_aa)
    joint_angles["right_ankle_roll"] = np.clip(-r_ankle_euler[0], np.radians(-25), np.radians(25))
    joint_angles["right_ankle"] = np.clip(r_ankle_euler[1], np.radians(-50), np.radians(50))
    r_foot_aa = pose[SMPL_JOINTS["r_foot"] * 3:(SMPL_JOINTS["r_foot"] + 1) * 3]
    r_foot_euler = axis_angle_to_euler(r_foot_aa)
    joint_angles["right_toe"] = np.clip(r_foot_euler[1], np.radians(-35), np.radians(55))

    # Left shoulder (2 DoF)
    l_shoulder_aa = pose[SMPL_JOINTS["l_shoulder"] * 3:(SMPL_JOINTS["l_shoulder"] + 1) * 3]
    l_shoulder_euler = axis_angle_to_euler(l_shoulder_aa)
    joint_angles["left_shoulder1"] = np.clip(l_shoulder_euler[0], np.radians(-85), np.radians(60))
    joint_angles["left_shoulder2"] = np.clip(l_shoulder_euler[1], np.radians(-85), np.radians(60))

    # Left elbow
    l_elbow_aa = pose[SMPL_JOINTS["l_elbow"] * 3:(SMPL_JOINTS["l_elbow"] + 1) * 3]
    l_elbow_euler = axis_angle_to_euler(l_elbow_aa)
    joint_angles["left_elbow"] = np.clip(l_elbow_euler[1], np.radians(-90), np.radians(50))

    # Right shoulder (2 DoF)
    r_shoulder_aa = pose[SMPL_JOINTS["r_shoulder"] * 3:(SMPL_JOINTS["r_shoulder"] + 1) * 3]
    r_shoulder_euler = axis_angle_to_euler(r_shoulder_aa)
    joint_angles["right_shoulder1"] = np.clip(r_shoulder_euler[0], np.radians(-60), np.radians(85))
    joint_angles["right_shoulder2"] = np.clip(r_shoulder_euler[1], np.radians(-60), np.radians(85))

    # Right elbow
    r_elbow_aa = pose[SMPL_JOINTS["r_elbow"] * 3:(SMPL_JOINTS["r_elbow"] + 1) * 3]
    r_elbow_euler = axis_angle_to_euler(r_elbow_aa)
    joint_angles["right_elbow"] = np.clip(r_elbow_euler[1], np.radians(-90), np.radians(50))

    # Head (2 DoF)
    head_aa = pose[SMPL_JOINTS["head"] * 3:(SMPL_JOINTS["head"] + 1) * 3]
    head_euler = axis_angle_to_euler(head_aa)
    joint_angles["head_yaw"] = np.clip(head_euler[2], np.radians(-50), np.radians(50))
    joint_angles["head_pitch"] = np.clip(head_euler[1], np.radians(-30), np.radians(30))

    for joint_offset, joint_name in enumerate(skeleton.joint_names):
        if joint_name not in joint_angles:
            raise KeyError(f"Missing retargeted joint angle for '{joint_name}'")
        qpos[7 + joint_offset] = joint_angles[joint_name]

    return qpos


def compute_qvel_from_qpos(
    qpos_t: np.ndarray,
    qpos_t1: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute qvel from consecutive qpos frames using finite differences.

    Args:
        qpos_t: qpos at time t
        qpos_t1: qpos at time t+1
        dt: Time step between frames

    Returns:
        qvel approximation
    """
    nv = len(qpos_t) - 1  # One less because quaternion -> angular velocity
    qvel = np.zeros(nv)

    # Root linear velocity
    qvel[0:3] = (qpos_t1[0:3] - qpos_t[0:3]) / dt

    # Root angular velocity (approximate from quaternion difference)
    # q_diff = q_t1 * q_t^-1
    q_t = qpos_t[3:7]
    q_t1 = qpos_t1[3:7]

    # Quaternion inverse: conj(q) / |q|^2, for unit quaternion just conjugate
    q_t_inv = np.array([q_t[0], -q_t[1], -q_t[2], -q_t[3]])

    # Quaternion multiply: q_diff = q_t1 * q_t_inv
    q_diff = quat_multiply(q_t1, q_t_inv)

    # Convert small rotation quaternion to angular velocity
    # For small angles: omega ≈ 2 * (x, y, z) / dt
    qvel[3:6] = 2.0 * q_diff[1:4] / dt

    # Joint velocities (simple finite difference)
    qvel[6:] = (qpos_t1[7:] - qpos_t[7:]) / dt

    return qvel


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z format).

    Args:
        q1, q2: Quaternions as (w, x, y, z)

    Returns:
        Product quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def retarget_amass_clip(
    poses: np.ndarray,
    trans: np.ndarray,
    fps: float,
    name: str,
    category: str,
    source_file: str,
    skeleton: Optional[SkeletonConfig] = None,
    height_offset: float = 0.93,  # Approximate humanoid standing height offset
    target_fps: Optional[float] = None,
) -> MotionClip:
    """Retarget an AMASS motion sequence to MuJoCo humanoid.

    Args:
        poses: SMPL pose array (T, 72) - T frames, 24 joints * 3 axis-angle
        trans: SMPL translation array (T, 3)
        fps: Original frame rate
        name: Clip name
        category: Motion category
        source_file: Source file path
        skeleton: Target skeleton config (default if None)
        height_offset: Height offset for ground clearance
        target_fps: Target fps for resampling (None to keep original)

    Returns:
        MotionClip with retargeted motion
    """
    if skeleton is None:
        skeleton = SkeletonConfig()

    T = len(poses)
    assert len(trans) == T, "poses and trans must have same length"

    # Resample if needed
    if target_fps is not None and target_fps != fps:
        # Simple nearest-neighbor resampling
        old_times = np.arange(T) / fps
        duration = old_times[-1]
        new_T = int(duration * target_fps) + 1
        new_times = np.arange(new_T) / target_fps

        # Find nearest frame for each new time
        indices = np.array([np.argmin(np.abs(old_times - t)) for t in new_times])

        poses = poses[indices]
        trans = trans[indices]
        fps = target_fps
        T = new_T

    # Retarget each frame
    qpos_seq = np.zeros((T, skeleton.nq))
    for t in range(T):
        qpos_seq[t] = retarget_smpl_frame(
            poses[t], trans[t], skeleton, height_offset
        )

    # Compute velocities from consecutive frames
    dt = 1.0 / fps
    qvel_seq = np.zeros((T, skeleton.nv))
    for t in range(T - 1):
        qvel_seq[t] = compute_qvel_from_qpos(qpos_seq[t], qpos_seq[t + 1], dt)
    # Last frame uses same velocity as previous
    qvel_seq[-1] = qvel_seq[-2] if T > 1 else np.zeros(skeleton.nv)

    return MotionClip(
        name=name,
        category=category,
        source_file=source_file,
        fps=fps,
        duration=(T - 1) / fps,
        qpos=qpos_seq.astype(np.float32),
        qvel=qvel_seq.astype(np.float32),
    )


def load_amass_npz(filepath: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load AMASS motion data from npz file.

    AMASS files typically contain:
    - 'poses': SMPL pose parameters (T, 156) for SMPL+H or (T, 72) for SMPL
    - 'trans': Root translation (T, 3)
    - 'mocap_framerate' or 'fps': Frame rate

    Args:
        filepath: Path to AMASS npz file

    Returns:
        Tuple of (poses, trans, fps)
    """
    data = np.load(filepath)

    # Get poses (handle SMPL vs SMPL+H)
    poses = data["poses"]
    if poses.shape[1] > 72:
        # SMPL+H format - take first 72 (body joints only)
        poses = poses[:, :72]

    trans = data["trans"]

    # Get frame rate
    if "mocap_framerate" in data:
        fps = float(data["mocap_framerate"])
    elif "fps" in data:
        fps = float(data["fps"])
    else:
        fps = 30.0  # Default assumption

    return poses, trans, fps
