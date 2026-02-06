"""Centralized experiment logging utilities built on Weights & Biases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


def _prefix_dict(data: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        out[f"{prefix}/{key}"] = value
    return out


@dataclass
class ExperimentLogger:
    """Structured experiment logger.

    Layout:
    - `common/*`: shared metrics across algorithms (sps, returns, success, etc.)
    - `algo/*`: algorithm-specific metrics
    - `task/*`: task-specific metrics
    - `extra/*`: optional custom metrics
    """

    _wandb: Optional[Any]
    _run: Optional[Any]

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        project: str,
        entity: Optional[str],
        run_name: Optional[str],
        group: Optional[str],
        tags: Optional[Sequence[str]],
        mode: str,
        run_dir: str,
        config: Mapping[str, Any],
    ) -> "ExperimentLogger":
        if not enabled:
            return cls(_wandb=None, _run=None)

        try:
            import wandb
        except Exception as exc:
            raise ImportError(
                "W&B logging requested but `wandb` is not installed. "
                "Install training dependencies with: pip install -e '.[train]'"
            ) from exc

        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=list(tags) if tags is not None else None,
            mode=mode,
            dir=run_dir,
            config=dict(config),
            save_code=False,
        )
        return cls(_wandb=wandb, _run=run)

    def log(
        self,
        *,
        step: int,
        common: Optional[Mapping[str, Any]] = None,
        algo: Optional[Mapping[str, Any]] = None,
        task: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._run is None:
            return

        payload: Dict[str, Any] = {}
        if common:
            payload.update(_prefix_dict(common, "common"))
        if algo:
            payload.update(_prefix_dict(algo, "algo"))
        if task:
            payload.update(_prefix_dict(task, "task"))
        if extra:
            payload.update(_prefix_dict(extra, "extra"))

        if payload:
            self._wandb.log(payload, step=int(step))

    def finish(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None
        self._wandb = None

