"""
training/extract_from_manifest.py
------------------------------------
Phase 7B (feature extraction step): processes exactly the 512 videos
listed in reports/dataset_split_manifest.json (built by
training/build_dataset_split.py, UNCHANGED by this file) -- the 3
confirmed duplicate copies are already excluded from that manifest, and
the train/test assignment for every video is read from it, not
recomputed here. This script creates NO new split.

Reuses, UNCHANGED:
    training/extract_dataset_features.py's _extract_features_isolated()
        -- runs pose estimation + angle/movement analysis for ONE video
        in its own forked subprocess, so that video's memory (which can
        reach several GB for a long/high-resolution clip -- see the
        note in that file) is fully released the instant it finishes,
        and a single video that exhausts memory only kills its own
        subprocess, not this whole batch run.
    Every pure computation module underneath that (pose_estimation.py,
        angle_calculator.py, the 4 movement modules, feature_builder.py)
        -- completely unmodified, same as every prior phase.

Every video is processed ONE AT A TIME (no batching, no parallel
workers) specifically because of the memory-scaling problem hit during
the earlier Phase 7B attempt -- see docs/ML_GUIDE.md.

Failure handling: if one video's subprocess fails for ANY reason
(timeout, OOM-kill, corrupt file, no person detected), it is logged
with its exact reason and the run CONTINUES to the next video. Nothing
is silently skipped -- every failure appears in both the console output
and the final JSON summary.

Output:
    dataset/extracted_features/train_features.csv
    dataset/extracted_features/test_features.csv
    reports/feature_extraction_report.json

Run:
    python training/extract_from_manifest.py
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from training.extract_dataset_features import RAW_VIDEOS_DIR, _extract_features_isolated

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_MANIFEST_PATH = os.path.join(BASE_DIR, "reports", "dataset_split_manifest.json")
TRAIN_OUTPUT_CSV = os.path.join(BASE_DIR, "dataset", "extracted_features", "train_features.csv")
TEST_OUTPUT_CSV = os.path.join(BASE_DIR, "dataset", "extracted_features", "test_features.csv")
EXTRACTION_REPORT_PATH = os.path.join(BASE_DIR, "reports", "feature_extraction_report.json")


def _load_manifest_video_list(manifest_path: str) -> List[Dict[str, str]]:
    """
    Flatten dataset_split_manifest.json's per-category train_files/
    test_files lists into one ordered list of
    {relative_path, video_type, quality_label, class_folder, split}
    dicts -- the exact 512 (406 train + 106 test) videos selected
    previously, nothing added or recomputed.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    videos = []
    for folder_name, info in manifest["categories"].items():
        for rel_path in info["train_files"]:
            videos.append(
                {
                    "relative_path": rel_path,
                    "video_type": info["video_type"],
                    "quality_label": info["quality_label"],
                    "class_folder": folder_name,
                    "split": "train",
                }
            )
        for rel_path in info["test_files"]:
            videos.append(
                {
                    "relative_path": rel_path,
                    "video_type": info["video_type"],
                    "quality_label": info["quality_label"],
                    "class_folder": folder_name,
                    "split": "test",
                }
            )
    return videos


def run_extraction(
    manifest_path: str = SPLIT_MANIFEST_PATH,
    raw_videos_dir: str = RAW_VIDEOS_DIR,
    train_output_csv: str = TRAIN_OUTPUT_CSV,
    test_output_csv: str = TEST_OUTPUT_CSV,
) -> Dict[str, object]:
    """
    Process every video in the manifest, one at a time in its own
    subprocess, and write two CSVs (train/test) preserving each row's
    quality_label and split assignment exactly as the manifest defined.
    """
    videos = _load_manifest_video_list(manifest_path)
    total = len(videos)

    train_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    start = time.time()
    for i, video in enumerate(videos, start=1):
        abs_path = os.path.join(raw_videos_dir, video["relative_path"])
        print(
            f"[{i}/{total}] ({video['split']}) Processing {video['relative_path']} ...",
            flush=True,
        )

        if not os.path.isfile(abs_path):
            failures.append(
                {
                    "file": video["relative_path"],
                    "split": video["split"],
                    "reason": "File listed in manifest not found on disk -- "
                    "the original 515-video set may have changed since the "
                    "manifest was built.",
                }
            )
            continue

        features, error = _extract_features_isolated(abs_path, video["video_type"])

        if error is not None:
            failures.append(
                {"file": video["relative_path"], "split": video["split"], "reason": error}
            )
            continue
        if features is None:
            failures.append(
                {
                    "file": video["relative_path"],
                    "split": video["split"],
                    "reason": "No person detected in any sampled frame",
                }
            )
            continue

        features["source_file"] = video["relative_path"]
        features["class_folder"] = video["class_folder"]
        features["quality_label"] = video["quality_label"]
        features["split"] = video["split"]

        if video["split"] == "train":
            train_rows.append(features)
        else:
            test_rows.append(features)

    os.makedirs(os.path.dirname(train_output_csv), exist_ok=True)
    train_df = pd.DataFrame(train_rows) if train_rows else pd.DataFrame()
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame()
    if not train_df.empty:
        train_df.to_csv(train_output_csv, index=False)
    if not test_df.empty:
        test_df.to_csv(test_output_csv, index=False)

    elapsed = round(time.time() - start, 1)

    # Per-category success counts, so a failure concentrated in one
    # category (rather than spread evenly) is easy to spot in the report.
    per_category: Dict[str, Dict[str, int]] = {}
    for video in videos:
        cat = video["class_folder"]
        per_category.setdefault(cat, {"train_expected": 0, "test_expected": 0, "train_ok": 0, "test_ok": 0})
        per_category[cat][f"{video['split']}_expected"] += 1
    for row in train_rows:
        per_category[row["class_folder"]]["train_ok"] += 1
    for row in test_rows:
        per_category[row["class_folder"]]["test_ok"] += 1

    return {
        "total_videos_in_manifest": total,
        "train_expected": sum(1 for v in videos if v["split"] == "train"),
        "test_expected": sum(1 for v in videos if v["split"] == "test"),
        "train_successful": len(train_rows),
        "test_successful": len(test_rows),
        "total_successful": len(train_rows) + len(test_rows),
        "total_failed": len(failures),
        "failures": failures,
        "per_category": per_category,
        "train_output_csv": train_output_csv,
        "test_output_csv": test_output_csv,
        "train_columns": list(train_df.columns) if not train_df.empty else [],
        "test_columns": list(test_df.columns) if not test_df.empty else [],
        "elapsed_seconds": elapsed,
    }


def main():
    if not os.path.isfile(SPLIT_MANIFEST_PATH):
        print(
            f"ERROR: {SPLIT_MANIFEST_PATH} not found. Run "
            "`python training/build_dataset_split.py` first."
        )
        sys.exit(1)

    result = run_extraction()

    os.makedirs(os.path.dirname(EXTRACTION_REPORT_PATH), exist_ok=True)
    with open(EXTRACTION_REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Done in {result['elapsed_seconds']}s")
    print(f"Train: {result['train_successful']} / {result['train_expected']} succeeded")
    print(f"Test:  {result['test_successful']} / {result['test_expected']} succeeded")
    print(f"Total failed: {result['total_failed']}")
    if result["failures"]:
        print("\nFailures:")
        for fail in result["failures"]:
            print(f"  [{fail['split']}] {fail['file']}: {fail['reason']}")
    print(f"\nTrain CSV: {result['train_output_csv']}")
    print(f"Test CSV:  {result['test_output_csv']}")
    print(f"Report:    {EXTRACTION_REPORT_PATH}")


if __name__ == "__main__":
    main()
