"""Abstract base class for task configurations."""

from abc import ABC, abstractmethod
from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np
import mujoco

from humanoid_collab.utils.ids import IDCache


class TaskConfig(ABC):
    """Abstract base class defining the interface for collaborative tasks.

    Each task must implement methods for:
    - MJCF additions (extra bodies/objects in the scene)
    - Geom group registration (for contact detection)
    - Contact pair definitions
    - Initial state randomization
    - Task-specific observations
    - Reward computation
    - Success condition checking
    - Fall/termination checking
    - Curriculum stage management
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Task name identifier."""
        ...

    @property
    @abstractmethod
    def num_curriculum_stages(self) -> int:
        """Number of curriculum stages available."""
        ...

    def mjcf_worldbody_additions(self) -> str:
        """Return XML fragment to inject into <worldbody>.

        Override this to add task-specific bodies (e.g., a box to lift).
        Default returns empty string (no additions).
        """
        return ""

    def mjcf_actuator_additions(self) -> str:
        """Return XML fragment to inject into <actuator>.

        Override this to add task-specific actuators.
        Default returns empty string.
        """
        return ""

    def register_geom_groups(self, model: mujoco.MjModel, id_cache: IDCache) -> None:
        """Register task-specific geom groups in the ID cache.

        Called after model compilation. Override to register groups for
        task-specific objects (e.g., box geoms).

        Args:
            model: Compiled MuJoCo model
            id_cache: ID cache to register groups into
        """
        pass

    @abstractmethod
    def get_contact_pairs(self) -> List[Tuple[str, str, str]]:
        """Return contact pairs to monitor.

        Returns:
            List of (result_key, group_a_name, group_b_name) tuples.
        """
        ...

    @abstractmethod
    def randomize_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        id_cache: IDCache,
        rng: np.random.RandomState,
    ) -> None:
        """Randomize initial state for this task.

        Args:
            model: MuJoCo model
            data: MuJoCo data (modify qpos/qvel in place)
            id_cache: ID cache for looking up indices
            rng: Random state for reproducibility
        """
        ...

    @property
    @abstractmethod
    def task_obs_dim(self) -> int:
        """Dimension of task-specific observations appended to base obs."""
        ...

    @abstractmethod
    def compute_task_obs(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
        agent: str,
        contact_info: Dict[str, bool],
    ) -> np.ndarray:
        """Compute task-specific observation components for one agent.

        Args:
            data: MuJoCo data
            id_cache: ID cache
            agent: Agent ID ('h0' or 'h1')
            contact_info: Contact detection results

        Returns:
            Float32 array of shape (task_obs_dim,)
        """
        ...

    @abstractmethod
    def compute_reward(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
        contact_info: Dict[str, bool],
        ctrl: np.ndarray,
        contact_force_proxy: float,
        hold_steps: int,
        success: bool,
        fallen: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute the reward for the current timestep.

        Both agents receive the same reward (cooperative).

        Returns:
            Tuple of (reward, info_dict)
        """
        ...

    @abstractmethod
    def check_success(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
        contact_info: Dict[str, bool],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if the task-specific success condition is met.

        Returns:
            Tuple of (condition_met, debug_info)
        """
        ...

    def check_fallen(
        self,
        data: mujoco.MjData,
        id_cache: IDCache,
    ) -> Tuple[bool, str]:
        """Check if either humanoid has fallen.

        Default implementation checks height and tilt.

        Returns:
            Tuple of (fallen, which_agent_or_empty)
        """
        from humanoid_collab.utils.kinematics import get_up_vector, compute_tilt_angle

        for agent in ["h0", "h1"]:
            xpos = id_cache.get_torso_xpos(data, agent)
            xmat = id_cache.get_torso_xmat(data, agent)
            up_vec = get_up_vector(xmat)
            tilt = compute_tilt_angle(up_vec)

            if xpos[2] < 0.5:
                return True, agent
            if tilt > np.pi / 2:
                return True, agent

        return False, ""

    @abstractmethod
    def set_stage(self, stage: int) -> None:
        """Set the curriculum stage."""
        ...

    @abstractmethod
    def get_weights_dict(self) -> Dict[str, float]:
        """Get current reward weights as a dictionary for logging."""
        ...
