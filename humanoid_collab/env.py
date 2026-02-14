"""PettingZoo Parallel environment for two-agent humanoid collaboration."""

from typing import Dict, Any, Optional, Tuple, List
import functools

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from humanoid_collab.mjcf_builder import build_mjcf
from humanoid_collab.tasks.registry import get_task
from humanoid_collab.utils.ids import IDCache
from humanoid_collab.utils.contacts import ContactDetector
from humanoid_collab.utils.obs import ObservationBuilder


class HumanoidCollabEnv(ParallelEnv):
    """Two-agent humanoid collaboration environment.

    A PettingZoo ParallelEnv where two humanoid agents collaborate on
    configurable tasks (hug, handshake, box_lift, etc.).

    Args:
        task: Task name (e.g., 'hug', 'handshake', 'box_lift')
        render_mode: 'human' or 'rgb_array'
        horizon: Maximum episode length in steps
        frame_skip: Number of physics steps per action
        hold_target: Consecutive success-condition steps for completion
        stage: Curriculum stage
        physics_profile: MuJoCo physics profile ("default", "balanced", "train_fast")
        fixed_standing: If True, use task-specific standing anchors (no locomotion).
        control_mode: "all" (18 actuators) or "arms_only" (6 arm actuators).
        observation_mode: "proprio" (vector), "rgb" (egocentric color image),
            or "gray" (egocentric grayscale image).
        obs_rgb_width: Visual observation width (used when observation_mode is rgb/gray).
        obs_rgb_height: Visual observation height (used when observation_mode is rgb/gray).
        emit_proprio_info: If True, attach vector proprio observations to infos as
            `proprio_obs` on reset/step.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "humanoid_collab_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        task: str = "hug",
        render_mode: Optional[str] = None,
        horizon: int = 1000,
        frame_skip: int = 5,
        hold_target: int = 30,
        stage: int = 0,
        physics_profile: str = "default",
        fixed_standing: bool = False,
        control_mode: str = "all",
        observation_mode: str = "proprio",
        obs_rgb_width: int = 84,
        obs_rgb_height: int = 84,
        emit_proprio_info: bool = False,
    ):
        super().__init__()

        self.task_name = task
        self.render_mode = render_mode
        self.horizon = horizon
        self.frame_skip = frame_skip
        self.hold_target = hold_target
        self.stage = stage
        self.physics_profile = physics_profile
        self.fixed_standing = bool(fixed_standing)
        self.control_mode = str(control_mode)
        self.observation_mode = str(observation_mode)
        self.obs_rgb_width = int(obs_rgb_width)
        self.obs_rgb_height = int(obs_rgb_height)
        self.emit_proprio_info = bool(emit_proprio_info)
        if self.control_mode not in {"all", "arms_only"}:
            raise ValueError(
                f"Unknown control_mode '{self.control_mode}'. "
                "Expected one of: all, arms_only."
            )
        if self.observation_mode not in {"proprio", "rgb", "gray"}:
            raise ValueError(
                f"Unknown observation_mode '{self.observation_mode}'. "
                "Expected one of: proprio, rgb, gray."
            )
        if self.obs_rgb_width <= 0 or self.obs_rgb_height <= 0:
            raise ValueError("obs_rgb_width and obs_rgb_height must be positive.")

        # Get task configuration
        self.task_config = get_task(task)
        self.task_config.set_stage(stage)
        self.task_config.configure(
            fixed_standing=self.fixed_standing,
            control_mode=self.control_mode,
        )

        spawn_half_distance, h1_faces_h0 = self._resolve_spawn_setup()

        # Build MJCF with task additions
        xml_str = build_mjcf(
            task_worldbody_additions=self.task_config.mjcf_worldbody_additions(),
            task_actuator_additions=self.task_config.mjcf_actuator_additions(),
            physics_profile=self.physics_profile,
            fixed_standing=self.fixed_standing,
            fixed_standing_mode=self._resolve_fixed_standing_mode(),
            spawn_half_distance=spawn_half_distance,
            h1_faces_h0=h1_faces_h0,
        )

        # Compile MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        # Initialize ID cache with standard humanoid groups
        self.id_cache = IDCache(self.model)

        # Register task-specific geom groups
        self.task_config.register_geom_groups(self.model, self.id_cache)

        # Initialize contact detector with task-configured pairs
        self.contact_detector = ContactDetector(
            self.id_cache,
            self.task_config.get_contact_pairs(),
        )

        # Initialize observation builder
        self.obs_builder = ObservationBuilder(
            self.id_cache,
            self.model,
            task_obs_dim=self.task_config.task_obs_dim,
        )

        # Agent setup
        self.agents = ["h0", "h1"]
        self.possible_agents = self.agents.copy()
        self._obs_camera_names = {agent: f"{agent}_ego" for agent in self.possible_agents}

        # Control indices
        self._control_actuator_idx = self._build_control_actuator_indices()

        # Dimensions
        self._action_dim = len(self._control_actuator_idx["h0"])
        self._obs_dim = self.obs_builder.get_obs_dim()
        if self.observation_mode in {"rgb", "gray"}:
            self._validate_obs_cameras()
            self._validate_visual_context_support()

        # Episode state
        self._step_count = 0
        self._hold_steps = 0
        self._terminated = False
        self._truncated = False
        self._rng = np.random.RandomState()

        # Rendering
        self._viewer = None
        self._render_context = None
        self._obs_renderers: Dict[str, mujoco.Renderer] = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        if self.observation_mode in {"rgb", "gray"}:
            channels = 3 if self.observation_mode == "rgb" else 1
            return spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_rgb_height, self.obs_rgb_width, channels),
                dtype=np.uint8,
            )
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._action_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Handle options
        if options is not None:
            if "stage" in options:
                self.stage = options["stage"]
                self.task_config.set_stage(self.stage)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Task-specific initial state randomization
        self.task_config.randomize_state(self.model, self.data, self.id_cache, self._rng)
        if self.fixed_standing:
            self._enforce_fixed_standing_roots()

        # Forward to compute derived quantities
        mujoco.mj_forward(self.model, self.data)

        # Settle physics for tasks with free objects (e.g., box_lift)
        # so objects come to rest before the episode starts
        if self.task_config.mjcf_worldbody_additions():
            self.data.ctrl[:] = 0.0
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)
            # Zero all velocities after settling
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        # Reset episode state
        self._step_count = 0
        self._hold_steps = 0
        self._terminated = False
        self._truncated = False
        self.agents = self.possible_agents.copy()

        # Build observations
        contact_info = self.contact_detector.detect_contacts(self.data)
        observations = self._build_full_observations(contact_info)
        proprio_obs = (
            self._build_proprio_observations(contact_info)
            if self.emit_proprio_info
            else None
        )

        infos = {
            agent: {
                "task": self.task_name,
                "stage": self.stage,
                "physics_profile": self.physics_profile,
                "fixed_standing": self.fixed_standing,
                "control_mode": self.control_mode,
                "observation_mode": self.observation_mode,
                "root_height": float(self._get_root_height(agent)),
                "weights": self.task_config.get_weights_dict(),
                **(
                    {"proprio_obs": proprio_obs[agent]}
                    if proprio_obs is not None
                    else {}
                ),
            }
            for agent in self.agents
        }

        return observations, infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if self._terminated or self._truncated:
            return self._get_terminal_returns()

        # Pack actions into control vector
        ctrl = np.zeros(self.model.nu)
        for agent in self.agents:
            if agent in actions:
                action = np.clip(actions[agent], -1.0, 1.0)
                idx = self._control_actuator_idx[agent]
                ctrl[idx] = action
        self.data.ctrl[:] = ctrl

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # NaN check
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return self._handle_nan_error()

        # Detect contacts
        contact_info = self.contact_detector.detect_contacts(self.data)
        contact_force_proxy = self.contact_detector.get_contact_force_proxy(self.data)

        # Check success condition
        success_condition, success_info = self.task_config.check_success(
            self.data, self.id_cache, contact_info
        )

        if success_condition:
            self._hold_steps += 1
        else:
            self._hold_steps = 0

        success = self._hold_steps >= self.hold_target

        # Check for fall
        fallen, fallen_agent = self.task_config.check_fallen(self.data, self.id_cache)

        # Compute reward
        reward, reward_info = self.task_config.compute_reward(
            self.data,
            self.id_cache,
            contact_info,
            ctrl,
            contact_force_proxy,
            self._hold_steps,
            success,
            fallen,
        )

        # Determine termination/truncation
        termination_reason = None
        if success:
            self._terminated = True
            termination_reason = "success"
        elif fallen:
            self._terminated = True
            termination_reason = f"fall_{fallen_agent}"
        elif self._step_count >= self.horizon:
            self._truncated = True
            termination_reason = "horizon"

        # Build observations
        observations = self._build_full_observations(contact_info)

        # Build rewards (cooperative: same for both)
        rewards = {agent: reward for agent in self.agents}

        terminations = {agent: self._terminated for agent in self.agents}
        truncations = {agent: self._truncated for agent in self.agents}
        proprio_obs = (
            self._build_proprio_observations(contact_info)
            if self.emit_proprio_info
            else None
        )

        infos = {}
        for agent in self.agents:
            infos[agent] = {
                "step": self._step_count,
                "hold_steps": self._hold_steps,
                "termination_reason": termination_reason,
                "task": self.task_name,
                "stage": self.stage,
                "physics_profile": self.physics_profile,
                "fixed_standing": self.fixed_standing,
                "control_mode": self.control_mode,
                "observation_mode": self.observation_mode,
                "root_height": float(self._get_root_height(agent)),
                **reward_info,
                **success_info,
                **{k: v for k, v in contact_info.items()},
                **(
                    {"proprio_obs": proprio_obs[agent]}
                    if proprio_obs is not None
                    else {}
                ),
            }

        if self._terminated or self._truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _build_full_observations(self, contact_info: Dict[str, bool]) -> Dict[str, np.ndarray]:
        """Build full observations = base obs + task-specific obs."""
        if self.observation_mode in {"rgb", "gray"}:
            return self._build_visual_observations()
        return self._build_proprio_observations(contact_info)

    def _build_proprio_observations(self, contact_info: Dict[str, bool]) -> Dict[str, np.ndarray]:
        """Build proprio observations with task-specific vector features."""

        base_obs = self.obs_builder.build_base_observations(self.data)

        observations = {}
        for agent in self.possible_agents:
            task_obs = self.task_config.compute_task_obs(
                self.data, self.id_cache, agent, contact_info
            )
            observations[agent] = np.concatenate([base_obs[agent], task_obs]).astype(np.float32)

        return observations

    def _build_visual_observations(self) -> Dict[str, np.ndarray]:
        """Build per-agent egocentric visual observations (RGB or grayscale)."""
        observations: Dict[str, np.ndarray] = {}
        for agent in self.possible_agents:
            renderer = self._obs_renderers.get(agent)
            if renderer is None:
                try:
                    renderer = mujoco.Renderer(self.model, self.obs_rgb_height, self.obs_rgb_width)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to initialize RGB observation renderer. "
                        "An offscreen OpenGL context is required for observation_mode='rgb'. "
                        "Try setting MUJOCO_GL=egl (Linux) or running with an active display."
                    ) from exc
                self._obs_renderers[agent] = renderer
            try:
                renderer.update_scene(self.data, camera=self._obs_camera_names[agent])
                frame = renderer.render()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to render visual observation frame. "
                    "Verify OpenGL/offscreen rendering support in this runtime."
                ) from exc
            frame_u8 = np.asarray(frame, dtype=np.uint8)
            if self.observation_mode == "gray":
                observations[agent] = self._rgb_to_gray(frame_u8)
            else:
                observations[agent] = frame_u8
        return observations

    @staticmethod
    def _rgb_to_gray(frame_u8: np.ndarray) -> np.ndarray:
        # ITU-R BT.601 luma transform.
        gray = (
            0.299 * frame_u8[..., 0].astype(np.float32)
            + 0.587 * frame_u8[..., 1].astype(np.float32)
            + 0.114 * frame_u8[..., 2].astype(np.float32)
        )
        gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)
        return gray[..., None]

    def _validate_obs_cameras(self) -> None:
        for agent in self.possible_agents:
            camera_name = self._obs_camera_names[agent]
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                raise ValueError(
                    f"Observation camera '{camera_name}' not found in model. "
                    "Expected egocentric camera for each agent."
                )

    def _validate_visual_context_support(self) -> None:
        ctx = None
        try:
            ctx = mujoco.GLContext(self.obs_rgb_width, self.obs_rgb_height)
            if hasattr(ctx, "make_current"):
                ctx.make_current()
        except Exception as exc:
            raise RuntimeError(
                "Visual observations (rgb/gray) require an offscreen OpenGL context. "
                "Try setting MUJOCO_GL=egl (Linux) or run with an active display."
            ) from exc
        finally:
            if ctx is not None and hasattr(ctx, "free"):
                try:
                    ctx.free()
                except Exception:
                    pass

    def _get_terminal_returns(self):
        if self.observation_mode in {"rgb", "gray"}:
            channels = 3 if self.observation_mode == "rgb" else 1
            zero = np.zeros((self.obs_rgb_height, self.obs_rgb_width, channels), dtype=np.uint8)
            obs = {agent: zero.copy() for agent in self.possible_agents}
            return (
                obs,
                {agent: 0.0 for agent in self.possible_agents},
                {agent: True for agent in self.possible_agents},
                {agent: self._truncated for agent in self.possible_agents},
                {agent: {} for agent in self.possible_agents},
            )
        return (
            {agent: np.zeros(self._obs_dim, dtype=np.float32) for agent in self.possible_agents},
            {agent: 0.0 for agent in self.possible_agents},
            {agent: True for agent in self.possible_agents},
            {agent: self._truncated for agent in self.possible_agents},
            {agent: {} for agent in self.possible_agents},
        )

    def _handle_nan_error(self):
        self._terminated = True
        self.agents = []
        if self.observation_mode in {"rgb", "gray"}:
            channels = 3 if self.observation_mode == "rgb" else 1
            zero = np.zeros((self.obs_rgb_height, self.obs_rgb_width, channels), dtype=np.uint8)
            return (
                {agent: zero.copy() for agent in self.possible_agents},
                {agent: -100.0 for agent in self.possible_agents},
                {agent: True for agent in self.possible_agents},
                {agent: False for agent in self.possible_agents},
                {agent: {"termination_reason": "nan_error"} for agent in self.possible_agents},
            )
        return (
            {agent: np.zeros(self._obs_dim, dtype=np.float32) for agent in self.possible_agents},
            {agent: -100.0 for agent in self.possible_agents},
            {agent: True for agent in self.possible_agents},
            {agent: False for agent in self.possible_agents},
            {agent: {"termination_reason": "nan_error"} for agent in self.possible_agents},
        )

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_human(self) -> None:
        if self._viewer is None:
            try:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer_is_passive = True
            except Exception:
                self._viewer_is_passive = False
                try:
                    if self._render_context is None:
                        self._render_context = mujoco.Renderer(self.model, 480, 640)
                    import warnings
                    if not hasattr(self, '_viewer_warning_shown'):
                        warnings.warn(
                            "Passive viewer unavailable. On macOS, run with 'mjpython' "
                            "instead of 'python3' for interactive rendering, "
                            "or use --mode video to save a video file."
                        )
                        self._viewer_warning_shown = True
                except Exception:
                    pass

        if self._viewer is not None and self._viewer_is_passive:
            self._viewer.sync()

    def _render_rgb_array(self) -> np.ndarray:
        width, height = 640, 480
        if self._render_context is None:
            self._render_context = mujoco.Renderer(self.model, height, width)
        self._render_context.update_scene(self.data, camera="h0_track")
        return self._render_context.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._render_context is not None:
            self._render_context.close()
            self._render_context = None
        for renderer in self._obs_renderers.values():
            renderer.close()
        self._obs_renderers = {}

    def state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _resolve_spawn_setup(self) -> Tuple[float, bool]:
        """Select initial spawn layout."""
        if not self.fixed_standing:
            return 1.0, False
        # For fixed-standing hand-only training, keep agents within reach and facing each other.
        if self.task_name == "handshake":
            return 0.4, True
        if self.task_name == "hug":
            # Start slightly closer so fixed-standing policies can reach wrap/contact early.
            return 0.15, True
        return 0.9, True

    def _build_control_actuator_indices(self) -> Dict[str, np.ndarray]:
        """Map each agent to the actuator indices controlled by RL."""
        control_idx: Dict[str, np.ndarray] = {}
        for agent in self.possible_agents:
            idx: List[int] = []
            for i in range(self.model.nu):
                act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if act_name is None or not act_name.startswith(f"{agent}_"):
                    continue
                if self.control_mode == "all":
                    idx.append(i)
                    continue
                if self._is_actuator_selected_for_arms_only(act_name):
                    idx.append(i)
            if len(idx) == 0:
                raise ValueError(
                    f"No actuators selected for agent '{agent}' under "
                    f"control_mode='{self.control_mode}'."
                )
            control_idx[agent] = np.asarray(idx, dtype=np.int32)
        if len(control_idx["h0"]) != len(control_idx["h1"]):
            raise ValueError("Control actuator dim mismatch between h0 and h1.")
        return control_idx

    def _get_root_height(self, agent: str) -> float:
        qpos_idx = self.id_cache.joint_qpos_idx[agent]
        return float(self.data.qpos[qpos_idx[2]])

    @staticmethod
    def _is_arm_actuator_name(act_name: str) -> bool:
        arm_tokens = (
            "_left_shoulder1",
            "_left_shoulder2",
            "_left_elbow",
            "_right_shoulder1",
            "_right_shoulder2",
            "_right_elbow",
        )
        return any(token in act_name for token in arm_tokens)

    @staticmethod
    def _is_abdomen_actuator_name(act_name: str) -> bool:
        return ("_abdomen_y" in act_name) or ("_abdomen_z" in act_name)

    def _is_actuator_selected_for_arms_only(self, act_name: str) -> bool:
        # For fixed-standing hug, allow abdomen control so torsos can lean/wrap
        # while lower body remains anchored.
        if (
            self.task_name == "hug"
            and self.fixed_standing
            and self.control_mode == "arms_only"
            and self._is_abdomen_actuator_name(act_name)
        ):
            return True
        return self._is_arm_actuator_name(act_name)

    def _resolve_fixed_standing_mode(self) -> str:
        if self.task_name == "hug":
            return "lower_body"
        return "torso"

    def _enforce_fixed_standing_roots(self) -> None:
        """Pin root state to model initial state for fixed-standing setups."""
        for agent in self.possible_agents:
            qpos_idx = self.id_cache.joint_qpos_idx[agent]
            qvel_idx = self.id_cache.joint_qvel_idx[agent]
            self.data.qpos[qpos_idx[0:7]] = self.model.qpos0[qpos_idx[0:7]]
            self.data.qvel[qvel_idx[0:6]] = 0.0

    @property
    def max_num_agents(self) -> int:
        return 2

    @property
    def num_agents(self) -> int:
        return len(self.agents)
