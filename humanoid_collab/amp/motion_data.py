"""Motion clip data structures and loading utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


@dataclass
class MotionClip:
    """A single retargeted motion clip ready for AMP training.

    All arrays are stored in the MuJoCo humanoid's coordinate frame,
    with joint ordering matching the model.

    Attributes:
        name: Unique identifier for this clip
        category: Motion category (standing, walking, reaching)
        source_file: Original AMASS file this came from
        fps: Frames per second
        duration: Total duration in seconds
        qpos: Joint positions at each frame (T, nq_agent)
        qvel: Joint velocities at each frame (T, nv_agent)
    """
    name: str
    category: str
    source_file: str
    fps: float
    duration: float
    qpos: np.ndarray  # (T, nq_agent)
    qvel: np.ndarray  # (T, nv_agent)

    def __post_init__(self):
        assert self.qpos.ndim == 2, "qpos must be 2D (T, nq)"
        assert self.qvel.ndim == 2, "qvel must be 2D (T, nv)"
        assert len(self.qpos) == len(self.qvel), "qpos and qvel must have same length"

    @property
    def num_frames(self) -> int:
        return len(self.qpos)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def get_frame(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get qpos and qvel at a specific frame index."""
        return self.qpos[idx].copy(), self.qvel[idx].copy()

    def get_transition(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get state transition (qpos_t, qvel_t, qpos_t1, qvel_t1).

        Args:
            idx: Frame index (0 to num_frames - 2)

        Returns:
            Tuple of (qpos_t, qvel_t, qpos_t+1, qvel_t+1)
        """
        assert 0 <= idx < self.num_frames - 1, f"Invalid transition index {idx}"
        return (
            self.qpos[idx].copy(),
            self.qvel[idx].copy(),
            self.qpos[idx + 1].copy(),
            self.qvel[idx + 1].copy(),
        )

    def sample_random_frame(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random frame from this clip."""
        idx = rng.integers(0, self.num_frames)
        return self.get_frame(idx)

    def sample_random_transition(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random state transition from this clip."""
        idx = rng.integers(0, self.num_frames - 1)
        return self.get_transition(idx)

    def save(self, path: Path) -> None:
        """Save clip to npz file."""
        np.savez_compressed(
            path,
            name=self.name,
            category=self.category,
            source_file=self.source_file,
            fps=self.fps,
            duration=self.duration,
            qpos=self.qpos,
            qvel=self.qvel,
        )

    @classmethod
    def load(cls, path: Path) -> "MotionClip":
        """Load clip from npz file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            name=str(data["name"]),
            category=str(data["category"]),
            source_file=str(data["source_file"]),
            fps=float(data["fps"]),
            duration=float(data["duration"]),
            qpos=data["qpos"],
            qvel=data["qvel"],
        )


@dataclass
class MotionDataset:
    """Collection of motion clips organized by category.

    Attributes:
        clips: List of all motion clips
        clips_by_category: Clips indexed by category name
        total_transitions: Total number of state transitions across all clips
    """
    clips: List[MotionClip] = field(default_factory=list)
    clips_by_category: Dict[str, List[MotionClip]] = field(default_factory=dict)
    _total_transitions: int = 0
    _category_weights: Optional[np.ndarray] = None

    def add_clip(self, clip: MotionClip) -> None:
        """Add a clip to the dataset."""
        self.clips.append(clip)

        if clip.category not in self.clips_by_category:
            self.clips_by_category[clip.category] = []
        self.clips_by_category[clip.category].append(clip)

        self._total_transitions += clip.num_frames - 1
        self._category_weights = None  # Invalidate cache

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    @property
    def categories(self) -> List[str]:
        return list(self.clips_by_category.keys())

    def get_category_weights(self) -> np.ndarray:
        """Get sampling weights proportional to transitions per category."""
        if self._category_weights is None:
            weights = []
            for cat in self.categories:
                cat_transitions = sum(c.num_frames - 1 for c in self.clips_by_category[cat])
                weights.append(cat_transitions)
            self._category_weights = np.array(weights) / sum(weights)
        return self._category_weights

    def sample_clip(
        self,
        rng: np.random.Generator,
        category: Optional[str] = None,
    ) -> MotionClip:
        """Sample a random clip, optionally from a specific category."""
        if category is not None:
            clips = self.clips_by_category.get(category, [])
            if not clips:
                raise ValueError(f"No clips in category '{category}'")
            return clips[rng.integers(0, len(clips))]

        # Sample category proportional to number of transitions
        weights = self.get_category_weights()
        cat_idx = rng.choice(len(self.categories), p=weights)
        category = self.categories[cat_idx]
        return self.sample_clip(rng, category=category)

    def sample_transition(
        self,
        rng: np.random.Generator,
        category: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random state transition from the dataset."""
        clip = self.sample_clip(rng, category=category)
        return clip.sample_random_transition(rng)

    def sample_transitions_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        category: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of state transitions.

        Returns:
            Tuple of (qpos_t, qvel_t, qpos_t1, qvel_t1) each with shape (batch_size, dim)
        """
        qpos_t_list = []
        qvel_t_list = []
        qpos_t1_list = []
        qvel_t1_list = []

        for _ in range(batch_size):
            qpos_t, qvel_t, qpos_t1, qvel_t1 = self.sample_transition(rng, category)
            qpos_t_list.append(qpos_t)
            qvel_t_list.append(qvel_t)
            qpos_t1_list.append(qpos_t1)
            qvel_t1_list.append(qvel_t1)

        return (
            np.stack(qpos_t_list),
            np.stack(qvel_t_list),
            np.stack(qpos_t1_list),
            np.stack(qvel_t1_list),
        )

    def save(self, directory: Path) -> None:
        """Save dataset to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "num_clips": self.num_clips,
            "total_transitions": self.total_transitions,
            "categories": self.categories,
            "clips": [
                {
                    "name": c.name,
                    "category": c.category,
                    "source_file": c.source_file,
                    "fps": c.fps,
                    "duration": c.duration,
                    "num_frames": c.num_frames,
                }
                for c in self.clips
            ],
        }

        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save each clip
        clips_dir = directory / "clips"
        clips_dir.mkdir(exist_ok=True)
        for clip in self.clips:
            clip.save(clips_dir / f"{clip.name}.npz")

    @classmethod
    def load(cls, directory: Path) -> "MotionDataset":
        """Load dataset from a directory."""
        directory = Path(directory)

        with open(directory / "metadata.json") as f:
            metadata = json.load(f)

        dataset = cls()
        clips_dir = directory / "clips"

        for clip_info in metadata["clips"]:
            clip_path = clips_dir / f"{clip_info['name']}.npz"
            clip = MotionClip.load(clip_path)
            dataset.add_clip(clip)

        return dataset


def load_motion_dataset(
    data_dir: str,
    categories: Optional[Tuple[str, ...]] = None,
) -> MotionDataset:
    """Load processed motion dataset from directory.

    Args:
        data_dir: Path to processed motion data directory
        categories: Optional filter for specific categories

    Returns:
        MotionDataset with loaded clips
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Motion data directory not found: {data_dir}")

    dataset = MotionDataset.load(data_path)

    if categories is not None:
        # Filter to requested categories
        filtered = MotionDataset()
        for clip in dataset.clips:
            if clip.category in categories:
                filtered.add_clip(clip)
        return filtered

    return dataset
