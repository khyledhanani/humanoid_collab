#!/usr/bin/env python3
"""AMASS download helper for AMP data preparation.

AMASS requires account authentication and license acceptance, so this helper
focuses on:
1) creating the expected local folder layout
2) printing the subset checklist to download manually
3) verifying that required subsets contain motion files

Usage:
    python scripts/download_amass.py --prepare
    python scripts/download_amass.py --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


AMASS_DATASET_URL = "https://amass.is.tue.mpg.de/"

# Core subsets used by the AMP pipeline.
DEFAULT_SUBSETS = ["CMU", "BMLrub", "ACCAD", "MPI_HDM05"]

# High-level notes only; authenticated direct links are user/session specific.
SUBSET_NOTES: Dict[str, str] = {
    "CMU": "standing + walking coverage",
    "BMLrub": "standing/idle coverage",
    "ACCAD": "walking + reaching coverage",
    "MPI_HDM05": "reaching/gesture coverage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or verify AMASS subset downloads.")
    parser.add_argument(
        "--amass-dir",
        type=str,
        default="data/amass/raw",
        help="Directory where AMASS subsets are stored",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=DEFAULT_SUBSETS,
        help="Subsets required for AMP training",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create subset directories and print download checklist",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify subset directories contain .npz motion files",
    )
    return parser.parse_args()


def prepare_layout(amass_dir: Path, subsets: List[str]) -> None:
    amass_dir.mkdir(parents=True, exist_ok=True)
    for subset in subsets:
        (amass_dir / subset).mkdir(parents=True, exist_ok=True)

    print(f"Created AMASS layout under: {amass_dir}")
    print("")
    print("Manual download checklist:")
    print(f"1. Open {AMASS_DATASET_URL}")
    print("2. Sign in and accept AMASS terms for each subset.")
    print("3. Download and extract each subset into its matching folder:")
    for subset in subsets:
        note = SUBSET_NOTES.get(subset, "subset used by your AMP config")
        print(f"   - {subset}: {note} -> {amass_dir / subset}")
    print("")
    print("Then run:")
    print(f"  python scripts/download_amass.py --verify --amass-dir {amass_dir}")
    print(
        f"  python scripts/process_amass.py --amass-dir {amass_dir} "
        "--output-dir data/amass/processed"
    )


def verify_layout(amass_dir: Path, subsets: List[str]) -> int:
    if not amass_dir.exists():
        print(f"AMASS directory does not exist: {amass_dir}")
        return 1

    missing = []
    empty = []

    print(f"Verifying AMASS subsets in: {amass_dir}")
    for subset in subsets:
        subset_dir = amass_dir / subset
        if not subset_dir.exists():
            missing.append(subset)
            continue

        npz_count = sum(1 for _ in subset_dir.rglob("*.npz"))
        if npz_count == 0:
            empty.append(subset)
        else:
            print(f"  {subset}: {npz_count} .npz files")

    if missing:
        print("")
        print("Missing subset directories:")
        for subset in missing:
            print(f"  - {subset}")

    if empty:
        print("")
        print("Subset directories with no .npz files:")
        for subset in empty:
            print(f"  - {subset}")

    if missing or empty:
        return 1

    print("")
    print("AMASS verification passed.")
    return 0


def main() -> None:
    args = parse_args()
    amass_dir = Path(args.amass_dir)

    # Default behavior: do both if neither flag is provided.
    run_prepare = args.prepare or (not args.prepare and not args.verify)
    run_verify = args.verify or (not args.prepare and not args.verify)

    if run_prepare:
        prepare_layout(amass_dir, args.subsets)

    if run_verify:
        code = verify_layout(amass_dir, args.subsets)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()

