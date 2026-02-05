"""Task configurations for the humanoid collaboration environment."""

from humanoid_collab.tasks.base import TaskConfig
from humanoid_collab.tasks.registry import TASK_REGISTRY, get_task

__all__ = ["TaskConfig", "TASK_REGISTRY", "get_task"]
