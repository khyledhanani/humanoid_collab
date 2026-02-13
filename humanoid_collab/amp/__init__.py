"""Adversarial Motion Priors (AMP) module for natural humanoid motion."""

from humanoid_collab.amp.amp_config import AMPConfig
from humanoid_collab.amp.motion_data import MotionClip, MotionDataset
from humanoid_collab.amp.amp_obs import AMPObsBuilder

__all__ = [
    "AMPConfig",
    "MotionClip",
    "MotionDataset",
    "AMPObsBuilder",
]

# Torch-dependent modules are optional so preprocessing utilities can run
# without installing training dependencies.
try:
    from humanoid_collab.amp.discriminator import AMPDiscriminator
    from humanoid_collab.amp.amp_reward import AMPRewardComputer
    from humanoid_collab.amp.motion_buffer import MotionReplayBuffer
except ImportError:
    AMPDiscriminator = None  # type: ignore[assignment]
    AMPRewardComputer = None  # type: ignore[assignment]
    MotionReplayBuffer = None  # type: ignore[assignment]
else:
    __all__.extend([
        "AMPDiscriminator",
        "AMPRewardComputer",
        "MotionReplayBuffer",
    ])
