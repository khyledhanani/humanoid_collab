"""Multiprocess vectorized runner for HumanoidCollabEnv."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from humanoid_collab.env import HumanoidCollabEnv


def _select_start_method(user_value: Optional[str]) -> str:
    if user_value is not None:
        return user_value
    methods = mp.get_all_start_methods()
    if "forkserver" in methods:
        return "forkserver"
    return "spawn"


def _worker(
    remote: Connection,
    parent_remote: Connection,
    env_kwargs: Dict[str, Any],
    auto_reset: bool,
) -> None:
    parent_remote.close()
    env = HumanoidCollabEnv(**env_kwargs)

    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                seed, options = payload
                obs, infos = env.reset(seed=seed, options=options)
                remote.send((obs, infos))
            elif cmd == "step":
                actions = payload
                if auto_reset and not env.agents:
                    env.reset(seed=None)

                obs, rewards, terminations, truncations, infos = env.step(actions)
                done = bool(terminations.get("h0", False) or truncations.get("h0", False))

                if auto_reset and done:
                    final_obs = obs
                    final_infos = infos
                    reset_obs, reset_infos = env.reset(seed=None)
                    merged_infos = {}
                    for agent in env.possible_agents:
                        agent_info = dict(final_infos.get(agent, {}))
                        agent_info["final_observation"] = final_obs.get(agent)
                        agent_info["reset_info"] = reset_infos.get(agent, {})
                        merged_infos[agent] = agent_info
                    obs = reset_obs
                    infos = merged_infos

                remote.send((obs, rewards, terminations, truncations, infos))
            elif cmd == "spaces":
                remote.send(
                    (
                        env.observation_space("h0"),
                        env.action_space("h0"),
                        env.possible_agents,
                    )
                )
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown command '{cmd}'")
    finally:
        try:
            env.close()
        except Exception:
            pass


_DTYPE_TO_CTYPE = {
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float64): ctypes.c_double,
    np.dtype(np.int32): ctypes.c_int32,
    np.dtype(np.int64): ctypes.c_int64,
    np.dtype(np.uint8): ctypes.c_uint8,
    np.dtype(np.bool_): ctypes.c_uint8,
}


def _create_shared_array(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    ctx: mp.context.BaseContext,
) -> Tuple[Any, np.ndarray]:
    dtype = np.dtype(dtype)
    ctype = _DTYPE_TO_CTYPE.get(dtype)
    if ctype is None:
        raise TypeError(f"Unsupported dtype for shared memory vec env: {dtype}")
    size = int(np.prod(shape))
    raw = ctx.RawArray(ctype, size)
    arr = np.frombuffer(raw, dtype=dtype, count=size).reshape(shape)
    return raw, arr


def _shared_spec_for_array(raw: Any, arr: np.ndarray) -> Dict[str, object]:
    return {
        "raw": raw,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
    }


def _attach_shared_array(spec: Dict[str, object]) -> Tuple[Any, np.ndarray]:
    shape = tuple(spec["shape"])  # type: ignore[arg-type]
    dtype = np.dtype(str(spec["dtype"]))
    raw = spec["raw"]
    size = int(np.prod(shape))
    arr = np.frombuffer(raw, dtype=dtype, count=size).reshape(shape)
    return raw, arr


def _sharedmem_worker(
    remote: Connection,
    parent_remote: Connection,
    env_kwargs: Dict[str, Any],
    auto_reset: bool,
    env_idx: int,
    shared_spec: Dict[str, object],
) -> None:
    parent_remote.close()
    env = HumanoidCollabEnv(**env_kwargs)

    possible_agents = list(env.possible_agents)

    try:
        action_bufs: Dict[str, np.ndarray] = {}
        obs_bufs: Dict[str, np.ndarray] = {}
        action_specs = shared_spec["actions"]  # type: ignore[assignment]
        obs_specs = shared_spec["observations"]  # type: ignore[assignment]

        for agent in possible_agents:
            _, arr = _attach_shared_array(action_specs[agent])  # type: ignore[index]
            action_bufs[agent] = arr

        for agent in possible_agents:
            _, arr = _attach_shared_array(obs_specs[agent])  # type: ignore[index]
            obs_bufs[agent] = arr

        _, reward_buf = _attach_shared_array(shared_spec["rewards"])  # type: ignore[arg-type]
        _, term_buf = _attach_shared_array(shared_spec["terminations"])  # type: ignore[arg-type]
        _, trunc_buf = _attach_shared_array(shared_spec["truncations"])  # type: ignore[arg-type]

        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                seed, options = payload
                obs, infos = env.reset(seed=seed, options=options)
                for agent in possible_agents:
                    obs_bufs[agent][env_idx] = np.asarray(obs[agent], dtype=obs_bufs[agent].dtype)
                remote.send(infos)
            elif cmd == "step":
                actions = {
                    agent: np.asarray(action_bufs[agent][env_idx], dtype=np.float32)
                    for agent in possible_agents
                }
                if auto_reset and not env.agents:
                    env.reset(seed=None)

                obs, rewards, terminations, truncations, infos = env.step(actions)
                done = bool(terminations.get("h0", False) or truncations.get("h0", False))

                if auto_reset and done:
                    final_obs = obs
                    final_infos = infos
                    reset_obs, reset_infos = env.reset(seed=None)
                    merged_infos = {}
                    for agent in possible_agents:
                        agent_info = dict(final_infos.get(agent, {}))
                        agent_info["final_observation"] = final_obs.get(agent)
                        agent_info["reset_info"] = reset_infos.get(agent, {})
                        merged_infos[agent] = agent_info
                    obs = reset_obs
                    infos = merged_infos

                for agent_idx, agent in enumerate(possible_agents):
                    obs_bufs[agent][env_idx] = np.asarray(obs[agent], dtype=obs_bufs[agent].dtype)
                    reward_buf[env_idx, agent_idx] = float(rewards[agent])
                    term_buf[env_idx, agent_idx] = bool(terminations[agent])
                    trunc_buf[env_idx, agent_idx] = bool(truncations[agent])

                remote.send(infos)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown command '{cmd}'")
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocHumanoidCollabVecEnv:
    """Multiprocess vectorized environment for HumanoidCollabEnv.

    Each environment instance runs in its own process.
    """

    def __init__(
        self,
        num_envs: int,
        env_kwargs: Optional[Dict[str, Any]] = None,
        auto_reset: bool = True,
        start_method: Optional[str] = None,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")

        self.num_envs = int(num_envs)
        self.env_kwargs = dict(env_kwargs or {})
        self.auto_reset = bool(auto_reset)
        self.start_method = _select_start_method(start_method)

        ctx = mp.get_context(self.start_method)
        self._remotes: List[Connection] = []
        self._work_remotes: List[Connection] = []
        self._processes: List[mp.Process] = []
        self._closed = False

        for _ in range(self.num_envs):
            work_remote, remote = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(work_remote, remote, self.env_kwargs, self.auto_reset),
                daemon=True,
            )
            proc.start()
            work_remote.close()
            self._remotes.append(remote)
            self._work_remotes.append(work_remote)
            self._processes.append(proc)

        self._remotes[0].send(("spaces", None))
        self._obs_space, self._act_space, self.possible_agents = self._remotes[0].recv()
        self.agents = list(self.possible_agents)
        self._waiting = False

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str):
        return self._act_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Dict[str, Any]]]]:
        for idx, remote in enumerate(self._remotes):
            child_seed = None if seed is None else int(seed + idx)
            remote.send(("reset", (child_seed, options)))
        results = [remote.recv() for remote in self._remotes]
        obs_list = [r[0] for r in results]
        infos_list = [r[1] for r in results]
        return self._stack_obs(obs_list), infos_list

    def step_async(self, actions: Dict[str, np.ndarray]) -> None:
        if self._waiting:
            raise RuntimeError("Already running step_async. Call step_wait() first.")

        for env_idx, remote in enumerate(self._remotes):
            per_env_actions = {
                agent: np.asarray(actions[agent][env_idx], dtype=np.float32)
                for agent in self.possible_agents
            }
            remote.send(("step", per_env_actions))
        self._waiting = True

    def step_wait(
        self,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        List[Dict[str, Dict[str, Any]]],
    ]:
        if not self._waiting:
            raise RuntimeError("step_wait() called without step_async()")
        results = [remote.recv() for remote in self._remotes]
        self._waiting = False

        obs_list = [r[0] for r in results]
        rewards_list = [r[1] for r in results]
        terms_list = [r[2] for r in results]
        truncs_list = [r[3] for r in results]
        infos_list = [r[4] for r in results]

        obs = self._stack_obs(obs_list)
        rewards = self._stack_scalar_dict(rewards_list, dtype=np.float32)
        terminations = self._stack_scalar_dict(terms_list, dtype=np.bool_)
        truncations = self._stack_scalar_dict(truncs_list, dtype=np.bool_)
        return obs, rewards, terminations, truncations, infos_list

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        List[Dict[str, Dict[str, Any]]],
    ]:
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        if self._closed:
            return

        if self._waiting:
            for remote in self._remotes:
                remote.recv()
            self._waiting = False

        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass

        for proc in self._processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()

        for remote in self._remotes:
            try:
                remote.close()
            except Exception:
                pass

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _stack_obs(obs_list: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        stacked = {}
        for agent in ["h0", "h1"]:
            stacked[agent] = np.stack(
                [np.asarray(obs[agent], dtype=np.float32) for obs in obs_list],
                axis=0,
            )
        return stacked

    @staticmethod
    def _stack_scalar_dict(
        data_list: Sequence[Dict[str, Any]],
        dtype,
    ) -> Dict[str, np.ndarray]:
        out = {}
        for agent in ["h0", "h1"]:
            out[agent] = np.asarray([d[agent] for d in data_list], dtype=dtype)
        return out


class SharedMemHumanoidCollabVecEnv:
    """Multiprocess vectorized environment using shared memory for step/reset tensors.

    Worker processes read actions and write observations/rewards/dones directly
    in shared memory. Pipes are used only for control signals and infos payloads.
    """

    def __init__(
        self,
        num_envs: int,
        env_kwargs: Optional[Dict[str, Any]] = None,
        auto_reset: bool = True,
        start_method: Optional[str] = None,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")

        self.num_envs = int(num_envs)
        self.env_kwargs = dict(env_kwargs or {})
        self.auto_reset = bool(auto_reset)
        self.start_method = _select_start_method(start_method)

        probe_env = HumanoidCollabEnv(**self.env_kwargs)
        try:
            self._obs_space = probe_env.observation_space("h0")
            self._act_space = probe_env.action_space("h0")
            self.possible_agents = list(probe_env.possible_agents)
        finally:
            probe_env.close()
        self.agents = list(self.possible_agents)

        if not isinstance(self._obs_space, spaces.Box):
            raise TypeError("SharedMemHumanoidCollabVecEnv requires Box observation spaces.")
        if not isinstance(self._act_space, spaces.Box):
            raise TypeError("SharedMemHumanoidCollabVecEnv requires Box action spaces.")

        self._obs_shape = tuple(self._obs_space.shape)
        self._obs_dtype = np.dtype(self._obs_space.dtype)
        self._act_shape = tuple(self._act_space.shape)
        self._act_dtype = np.float32
        if len(self._act_shape) != 1:
            raise ValueError("SharedMemHumanoidCollabVecEnv expects 1D continuous action spaces.")
        self._act_dim = int(self._act_shape[0])
        self._num_agents = len(self.possible_agents)

        self._action_raw: Dict[str, Any] = {}
        self._obs_raw: Dict[str, Any] = {}
        self._action_buf: Dict[str, np.ndarray] = {}
        self._obs_buf: Dict[str, np.ndarray] = {}
        self._reward_raw: Optional[Any] = None
        self._term_raw: Optional[Any] = None
        self._trunc_raw: Optional[Any] = None

        self._rewards = np.zeros((self.num_envs, self._num_agents), dtype=np.float32)
        self._terminations = np.zeros((self.num_envs, self._num_agents), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs, self._num_agents), dtype=np.bool_)
        self._ctx = mp.get_context(self.start_method)
        self._closed = False
        self._waiting = False
        self._remotes: List[Connection] = []
        self._work_remotes: List[Connection] = []
        self._processes: List[mp.Process] = []

        try:
            shared_spec = self._allocate_shared_buffers()

            for env_idx in range(self.num_envs):
                work_remote, remote = self._ctx.Pipe()
                proc = self._ctx.Process(
                    target=_sharedmem_worker,
                    args=(
                        work_remote,
                        remote,
                        self.env_kwargs,
                        self.auto_reset,
                        env_idx,
                        shared_spec,
                    ),
                    daemon=True,
                )
                proc.start()
                work_remote.close()
                self._remotes.append(remote)
                self._work_remotes.append(work_remote)
                self._processes.append(proc)
        except Exception:
            for remote in self._remotes:
                try:
                    remote.close()
                except Exception:
                    pass
            for proc in self._processes:
                if proc.is_alive():
                    proc.terminate()
            self._cleanup_shared_memory()
            raise

    def _allocate_shared_buffers(self) -> Dict[str, object]:
        action_specs: Dict[str, Dict[str, object]] = {}
        obs_specs: Dict[str, Dict[str, object]] = {}
        for agent in self.possible_agents:
            act_raw, act_arr = _create_shared_array((self.num_envs, self._act_dim), self._act_dtype, self._ctx)
            self._action_raw[agent] = act_raw
            self._action_buf[agent] = act_arr
            action_specs[agent] = _shared_spec_for_array(act_raw, act_arr)

            obs_raw, obs_arr = _create_shared_array((self.num_envs, *self._obs_shape), self._obs_dtype, self._ctx)
            self._obs_raw[agent] = obs_raw
            self._obs_buf[agent] = obs_arr
            obs_specs[agent] = _shared_spec_for_array(obs_raw, obs_arr)

        self._reward_raw, self._rewards = _create_shared_array((self.num_envs, self._num_agents), np.float32, self._ctx)
        self._term_raw, self._terminations = _create_shared_array((self.num_envs, self._num_agents), np.bool_, self._ctx)
        self._trunc_raw, self._truncations = _create_shared_array((self.num_envs, self._num_agents), np.bool_, self._ctx)
        self._rewards.fill(0.0)
        self._terminations.fill(False)
        self._truncations.fill(False)

        return {
            "actions": action_specs,
            "observations": obs_specs,
            "rewards": _shared_spec_for_array(self._reward_raw, self._rewards),
            "terminations": _shared_spec_for_array(self._term_raw, self._terminations),
            "truncations": _shared_spec_for_array(self._trunc_raw, self._truncations),
        }

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str):
        return self._act_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Dict[str, Any]]]]:
        for idx, remote in enumerate(self._remotes):
            child_seed = None if seed is None else int(seed + idx)
            remote.send(("reset", (child_seed, options)))
        infos_list = [remote.recv() for remote in self._remotes]
        return self._copy_obs(), infos_list

    def step_async(self, actions: Dict[str, np.ndarray]) -> None:
        if self._waiting:
            raise RuntimeError("Already running step_async. Call step_wait() first.")

        for agent in self.possible_agents:
            arr = np.asarray(actions[agent], dtype=self._act_dtype)
            expected = (self.num_envs, self._act_dim)
            if arr.shape != expected:
                raise ValueError(
                    f"Expected actions['{agent}'] shape {expected}, got {tuple(arr.shape)}"
                )
            self._action_buf[agent][:] = arr

        for remote in self._remotes:
            remote.send(("step", None))
        self._waiting = True

    def step_wait(
        self,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        List[Dict[str, Dict[str, Any]]],
    ]:
        if not self._waiting:
            raise RuntimeError("step_wait() called without step_async()")
        infos_list = [remote.recv() for remote in self._remotes]
        self._waiting = False

        obs = self._copy_obs()
        rewards = self._copy_scalars(self._rewards, dtype=np.float32)
        terminations = self._copy_scalars(self._terminations, dtype=np.bool_)
        truncations = self._copy_scalars(self._truncations, dtype=np.bool_)
        return obs, rewards, terminations, truncations, infos_list

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        List[Dict[str, Dict[str, Any]]],
    ]:
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return

        if self._waiting:
            for remote in self._remotes:
                remote.recv()
            self._waiting = False

        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass

        for proc in self._processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()

        for remote in self._remotes:
            try:
                remote.close()
            except Exception:
                pass

        self._cleanup_shared_memory()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _copy_obs(self) -> Dict[str, np.ndarray]:
        out = {}
        for agent in self.possible_agents:
            out[agent] = np.asarray(self._obs_buf[agent]).copy()
        return out

    def _copy_scalars(self, arr: np.ndarray, dtype) -> Dict[str, np.ndarray]:
        out = {}
        for idx, agent in enumerate(self.possible_agents):
            out[agent] = np.asarray(arr[:, idx], dtype=dtype).copy()
        return out

    def _cleanup_shared_memory(self) -> None:
        self._action_raw.clear()
        self._obs_raw.clear()
        self._action_buf.clear()
        self._obs_buf.clear()
        self._reward_raw = None
        self._term_raw = None
        self._trunc_raw = None
