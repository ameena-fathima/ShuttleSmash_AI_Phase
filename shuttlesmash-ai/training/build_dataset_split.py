"""
training/build_dataset_split.py
-----------------------------------
Builds the train/test split manifest used by the rest of the Phase 7
pipeline (training/extract_dataset_features.py and
training/train_dataset_model.py, both UNCHANGED by this file).

Two responsibilities, both file-system read-only -- nothing under
dataset/raw_videos/ is ever modified, renamed, or deleted:

1. Duplicate exclusion: reuses training/extract_dataset_features.py's
   verify_dataset() (unchanged) to find exact-duplicate videos (same
   MD5 content hash), and keeps exactly one file per duplicate group
   for the ML pipeline -- the excluded copy is a training-set
   *exclusion*, not a filesystem deletion; it stays on disk untouched.
   This matters specifically to prevent the same clip appearing in both
   the training and test split, which would leak test-set information
   into training and inflate reported accuracy.

2. Stratified 80/20 split: splits the surviving videos into train/test
   sets independently within each of the 8 class folders (so every
   category keeps its ~80/20 proportion), using scikit-learn's
   train_test_split with a fixed random_state for reproducibility.

Output:
    reports/dataset_split_manifest.json -- the exact list of video
    paths assigned to train vs. test, plus per-category counts. This
    manifest is what a later feature-extraction/training run should
    read from, so the exact same split is used consistently.

Run:
    python training/build_dataset_split.py
"""

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

from training.extract_dataset_features import (
    RAW_VIDEOS_DIR,
    _discover_dataset,
    _parse_class_folder_name,
    verify_dataset,
)

SPLIT_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "dataset_split_manifest.json"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


def _select_duplicate_exclusions(duplicate_files: Dict[str, List[str]]) -> List[str]:
    """
    For each group of exact-duplicate files (same content hash), keep
    the first one (sorted, so this is deterministic/reproducible) and
    return the rest as paths to EXCLUDE from the ML dataset. Excluding
    here only affects which videos this script assigns to train/test --
    it never touches the actual files on disk.
    """
    excluded = []
    for group in duplicate_files.values():
        keep, *drop = sorted(group)
        excluded.extend(drop)
    return excluded


def build_split(raw_videos_dir: str = RAW_VIDEOS_DIR) -> Dict[str, object]:
    """
    Compute the deduplicated, stratified train/test split.

    Returns a dict with per-category train/test file lists and counts,
    plus the overall totals -- the same structure written to
    reports/dataset_split_manifest.json.
    """
    verification = verify_dataset(raw_videos_dir)
    excluded_relative = set(_select_duplicate_exclusions(verification["duplicate_files"]))

    dataset = _discover_dataset(raw_videos_dir)

    categories: Dict[str, Dict[str, object]] = {}
    total_before = 0
    total_after = 0
    total_train = 0
    total_test = 0

    for folder_name, video_paths in sorted(dataset.items()):
        parsed = _parse_class_folder_name(folder_name)
        if parsed is None:
            continue
        quality, movement = parsed

        total_before += len(video_paths)

        kept_paths = []
        excluded_paths = []
        for path in video_paths:
            rel_path = os.path.relpath(path, raw_videos_dir)
            if rel_path in excluded_relative:
                excluded_paths.append(rel_path)
            else:
                kept_paths.append(rel_path)

        kept_paths = sorted(kept_paths)
        total_after += len(kept_paths)

        # Stratification note: within a single class folder every video
        # already shares the same label, so there's nothing further to
        # stratify BY inside one folder -- "stratified by category"
        # means splitting each of the 8 folders independently (this
        # loop), rather than shuffling all 512 videos together, which
        # is what actually guarantees every category keeps its ~80/20
        # proportion instead of some categories ending up over/under-
        # represented in the test set by chance.
        train_paths, test_paths = train_test_split(
            kept_paths, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        train_paths, test_paths = sorted(train_paths), sorted(test_paths)

        total_train += len(train_paths)
        total_test += len(test_paths)

        categories[folder_name] = {
            "quality_label": quality,
            "video_type": movement,
            "total_before_dedup": len(video_paths),
            "excluded_duplicates": excluded_paths,
            "total_after_dedup": len(kept_paths),
            "train_count": len(train_paths),
            "test_count": len(test_paths),
            "train_files": train_paths,
            "test_files": test_paths,
        }

    return {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "total_videos_before_dedup": total_before,
        "total_duplicate_exclusions": len(excluded_relative),
        "total_videos_after_dedup": total_after,
        "total_train": total_train,
        "total_test": total_test,
        "categories": categories,
    }


def main():
    result = build_split()
    os.makedirs(os.path.dirname(SPLIT_MANIFEST_PATH), exist_ok=True)
    with open(SPLIT_MANIFEST_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Total videos before dedup exclusion: {result['total_videos_before_dedup']}")
    print(f"Duplicate copies excluded from ML split: {result['total_duplicate_exclusions']}")
    print(f"Total videos used for ML split: {result['total_videos_after_dedup']}")
    print()
    print(f"{'Category':<16} {'Good/Poor':<10} {'Type':<10} {'Total':>6} {'Train':>6} {'Test':>6}")
    for folder_name, info in sorted(result["categories"].items()):
        print(
            f"{folder_name:<16} {info['quality_label']:<10} {info['video_type']:<10} "
            f"{info['total_after_dedup']:>6} {info['train_count']:>6} {info['test_count']:>6}"
        )
    print()
    print(f"TOTAL: {result['total_videos_after_dedup']} videos -> "
          f"{result['total_train']} train / {result['total_test']} test")
    print(f"\nManifest written to {SPLIT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
