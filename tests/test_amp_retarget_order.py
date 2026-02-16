"""Regression tests for AMP retargeting layout assumptions."""

import mujoco

from humanoid_collab import HumanoidCollabEnv
from humanoid_collab.amp.retarget import SkeletonConfig


def _agent_joint_order_from_model(model: mujoco.MjModel, agent: str) -> tuple[str, ...]:
    prefix = f"{agent}_"
    joint_names = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None or not name.startswith(prefix):
            continue
        if name == f"{agent}_root":
            continue
        joint_names.append(name[len(prefix):])
    return tuple(joint_names)


def test_retarget_joint_order_matches_model_qpos_order() -> None:
    env = HumanoidCollabEnv(task="hug", stage=0)
    try:
        model_order = _agent_joint_order_from_model(env.model, "h0")
    finally:
        env.close()

    assert model_order == SkeletonConfig().joint_names
