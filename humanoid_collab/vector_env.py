"""Multiprocess vectorized runner for HumanoidCollabEnv."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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

