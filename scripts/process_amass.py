#!/usr/bin/env python3
"""Process AMASS motion capture data for AMP training.

This script downloads and processes AMASS motion data, retargeting it
to the MuJoCo humanoid skeleton used in the environment.

Usage:
    python scripts/process_amass.py --amass-dir /path/to/amass --output-dir data/amass/processed

AMASS data can be downloaded from: https://amass.is.tue.mpg.de/
You need to download the following subsets:
- CMU (for walking, standing)
- BMLrub (for standing poses)
- ACCAD (for reaching motions)
- MPI_HDM05 (for reaching/gesture motions)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from humanoid_collab.amp.retarget import (
    retarget_amass_clip,
    load_amass_npz,
    SkeletonConfig,
)
from humanoid_collab.amp.motion_data import MotionClip, MotionDataset


# Motion category keywords for automatic classification
MOTION_CATEGORIES: Dict[str, List[str]] = {
    "standing": [
        "stand", "idle", "wait", "pause", "rest", "neutral",
        "t-pose", "a-pose", "tpose", "apose",
    ],
    "walking": [
        "walk", "locomotion", "step", "march", "stroll",
        "forward", "backward", "strafe", "side",
    ],
    "reaching": [
        "reach", "grab", "pick", "point", "wave", "gesture",
        "throw", "catch", "hand", "arm",
    ],
}


def classify_motion(filename: str) -> Optional[str]:
    """Classify motion category based on filename.

    Args:
        filename: Motion file name

    Returns:
        Category name or None if unclassified
    """
    filename_lower = filename.lower()

    for category, keywords in MOTION_CATEGORIES.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return category

    return None


def find_amass_files(
    amass_dir: Path,
    subsets: Optional[List[str]] = None,
) -> List[Tuple[Path, str]]:
    """Find all AMASS npz files in the given directory.

    Args:
        amass_dir: Path to AMASS data root
        subsets: Optional list of subset names to include

    Returns:
        List of (file_path, subset_name) tuples
    """
    files = []

    # If subsets specified, only search those directories
    if subsets:
        search_dirs = [amass_dir / s for s in subsets if (amass_dir / s).exists()]
    else:
        search_dirs = [amass_dir]

    for search_dir in search_dirs:
        for npz_file in search_dir.rglob("*.npz"):
            # Get subset name from path
            rel_path = npz_file.relative_to(amass_dir)
            subset = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
            files.append((npz_file, subset))

    return files


def process_single_file(
    filepath: Path,
    subset: str,
    skeleton: SkeletonConfig,
    target_fps: float = 30.0,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    height_offset: float = 0.93,
) -> Optional[MotionClip]:
    """Process a single AMASS file.

    Args:
        filepath: Path to AMASS npz file
        subset: Dataset subset name
        skeleton: Target skeleton configuration
        target_fps: Target frame rate
        min_duration: Minimum clip duration (seconds)
        max_duration: Maximum clip duration (seconds)
        height_offset: Height offset for ground clearance

    Returns:
        MotionClip or None if processing failed
    """
    try:
        # Load AMASS data
        poses, trans, fps = load_amass_npz(filepath)

        # Check duration
        duration = len(poses) / fps
        if duration < min_duration:
            return None
        if duration > max_duration:
            # Truncate to max duration
            max_frames = int(max_duration * fps)
            poses = poses[:max_frames]
            trans = trans[:max_frames]

        # Classify motion category
        category = classify_motion(filepath.stem)
        if category is None:
            category = "other"

        # Generate unique name
        name = f"{subset}_{filepath.stem}"
        name = name.replace(" ", "_").replace("-", "_")

        # Retarget motion
        clip = retarget_amass_clip(
            poses=poses,
            trans=trans,
            fps=fps,
            name=name,
            category=category,
            source_file=str(filepath),
            skeleton=skeleton,
            height_offset=height_offset,
            target_fps=target_fps,
        )

        return clip

    except Exception as e:
        print(f"  Error processing {filepath}: {e}")
        return None


def process_amass(
    amass_dir: str,
    output_dir: str,
    subsets: Optional[List[str]] = None,
    target_fps: float = 30.0,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    height_offset: float = 0.93,
    max_clips: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> MotionDataset:
    """Process AMASS motion data for AMP training.

    Args:
        amass_dir: Path to AMASS data root
        output_dir: Path to output directory
        subsets: Dataset subsets to process (None for all)
        target_fps: Target frame rate for resampling
        min_duration: Minimum clip duration (seconds)
        max_duration: Maximum clip duration (seconds)
        height_offset: Height offset for ground clearance
        max_clips: Maximum number of clips to process (None for all)
        categories: Only include these motion categories (None for all)

    Returns:
        Processed MotionDataset
    """
    amass_path = Path(amass_dir)
    output_path = Path(output_dir)

    if not amass_path.exists():
        raise FileNotFoundError(f"AMASS directory not found: {amass_dir}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize skeleton config
    skeleton = SkeletonConfig()

    # Find all AMASS files
    print(f"Searching for AMASS files in {amass_dir}...")
    files = find_amass_files(amass_path, subsets)
    print(f"Found {len(files)} files")

    if max_clips:
        files = files[:max_clips]

    # Process files
    dataset = MotionDataset()
    processed = 0
    skipped = 0
    errors = 0

    for i, (filepath, subset) in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(files)}...")

        clip = process_single_file(
            filepath=filepath,
            subset=subset,
            skeleton=skeleton,
            target_fps=target_fps,
            min_duration=min_duration,
            max_duration=max_duration,
            height_offset=height_offset,
        )

        if clip is None:
            skipped += 1
            continue

        # Filter by category if specified
        if categories is not None and clip.category not in categories:
            skipped += 1
            continue

        dataset.add_clip(clip)
        processed += 1

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total transitions: {dataset.total_transitions}")

    # Print category breakdown
    print("\nCategory breakdown:")
    for category in dataset.categories:
        cat_clips = dataset.clips_by_category[category]
        cat_transitions = sum(c.num_frames - 1 for c in cat_clips)
        print(f"  {category}: {len(cat_clips)} clips, {cat_transitions} transitions")

    # Save dataset
    print(f"\nSaving dataset to {output_dir}...")
    dataset.save(output_path)
    print("Done!")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Process AMASS motion capture data for AMP training"
    )
    parser.add_argument(
        "--amass-dir",
        type=str,
        required=True,
        help="Path to AMASS data root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/amass/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["CMU", "BMLrub", "ACCAD", "MPI_HDM05"],
        help="AMASS subsets to process",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target frame rate for resampling",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum clip duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum clip duration in seconds",
    )
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0.93,
        help="Height offset for ground clearance",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Maximum number of clips to process",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Only include these motion categories",
    )

    args = parser.parse_args()

    process_amass(
        amass_dir=args.amass_dir,
        output_dir=args.output_dir,
        subsets=args.subsets,
        target_fps=args.target_fps,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        height_offset=args.height_offset,
        max_clips=args.max_clips,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
