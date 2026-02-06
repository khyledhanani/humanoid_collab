"""Humanoid Collab: Two-agent humanoid collaboration environment with multiple tasks."""

from humanoid_collab.env import HumanoidCollabEnv
from humanoid_collab.mjx_env import MJXHumanoidCollabEnv
from humanoid_collab.vector_env import SubprocHumanoidCollabVecEnv

__all__ = ["HumanoidCollabEnv", "MJXHumanoidCollabEnv", "SubprocHumanoidCollabVecEnv"]
__version__ = "0.1.0"
