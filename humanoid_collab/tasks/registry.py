"""Task registry for the humanoid collaboration environment."""

from typing import Dict, Type
from humanoid_collab.tasks.base import TaskConfig


# Registry mapping task name to task config class
_TASK_CLASSES: Dict[str, Type[TaskConfig]] = {}


def register_task(cls: Type[TaskConfig]) -> Type[TaskConfig]:
    """Register a task config class."""
    instance = cls()
    _TASK_CLASSES[instance.name] = cls
    return cls


def get_task(name: str) -> TaskConfig:
    """Get a task config instance by name.

    Args:
        name: Task name (e.g., 'hug', 'handshake', 'box_lift')

    Returns:
        TaskConfig instance

    Raises:
        ValueError: If task name is not registered
    """
    if name not in _TASK_CLASSES:
        available = ", ".join(sorted(_TASK_CLASSES.keys()))
        raise ValueError(f"Unknown task '{name}'. Available tasks: {available}")
    return _TASK_CLASSES[name]()


def available_tasks():
    """Return list of available task names."""
    return sorted(_TASK_CLASSES.keys())


# Import task modules to trigger registration
from humanoid_collab.tasks import hug  # noqa: E402, F401
from humanoid_collab.tasks import handshake  # noqa: E402, F401
from humanoid_collab.tasks import box_lift  # noqa: E402, F401

TASK_REGISTRY = _TASK_CLASSES
