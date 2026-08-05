"""
training/extract_dataset_features.py
---------------------------------------
Phase 7A: converts a labeled badminton video dataset
(dataset/raw_videos/<ClassFolder>/*.mp4|.mkv|.ts|...) into an ML-ready
feature CSV, running each video through the SAME pose-estimation,
joint-angle, and movement-analysis pipeline the live app already uses
for every uploaded video -- just orchestrated as a standalone batch
script, since these dataset videos don't go through the web upload form.

Reuses, UNCHANGED (no logic duplicated or modified):
    utils/video_utils.py         (metadata + frame sampling)
    modules/pose_estimation.py   (PoseEstimator)
    modules/angle_calculator.py  (per-frame + aggregate joint angles)
    modules/footwork_analysis.py / smash_analysis.py / jump_analysis.py / lunge_analysis.py
    modules/feature_builder.py   (the same flat feature-row shape used
                                   everywhere else in the app)

Does NOT touch the Flask app, the `users`/`videos`/`analyses` tables,
or any existing route/service/module -- this is a read-only, offline
batch tool over dataset/raw_videos/. It never writes to, renames, or
deletes any of the original video files.

Expected folder layout (class folder name = "<Quality>_<MovementType>",
matched case-insensitively so a naming slip like "poor_Jump" still
works, though it's flagged in the verification report):
    dataset/raw_videos/
        Good_Footwork/*.mp4
        Poor_Footwork/*.mp4
        Good_Jump/*.mp4, *.mkv
        Poor_Jump/*.mp4           (or poor_Jump)
        Good_Lunge/*.ts, *.mkv
        Poor_Lunge/*.ts
        Good_Smash/*.mp4
        Poor_Smash/*.mp4

Output:
    dataset/extracted_features/labeled_dataset_features.csv
    reports/dataset_verification_report.json

Run:
    python training/extract_dataset_features.py
"""

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules import angle_calculator, feature_builder
from modules import footwork_analysis, jump_analysis, lunge_analysis, smash_analysis
from utils import video_utils
# PoseEstimator is imported lazily, inside _extract_features_for_video(),
# not here at module level. Dataset verification (Part 1-3 below) is a
# pure file-system/metadata check and has no pose-estimation dependency
# -- it should be runnable even before mediapipe is fully set up. Only
# the actual feature-extraction step (Part 4-5) needs it.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEOS_DIR = os.path.join(BASE_DIR, "dataset", "raw_videos")
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset", "extracted_features", "labeled_dataset_features.csv")
VERIFICATION_REPORT_PATH = os.path.join(BASE_DIR, "reports", "dataset_verification_report.json")

# Matches config.py's FRAME_SAMPLE_RATE, so a video analyzed here and
# the same video re-analyzed through the live app would sample the same
# frames.
FRAME_SAMPLE_RATE = 5

# This offline script reads video files directly with OpenCV (via
# utils/video_utils.py), which can decode more formats than the live
# web upload form allows. config.py's ALLOWED_VIDEO_EXTENSIONS
# (.mp4/.avi/.mov) is a UX/policy whitelist for the upload FORM, not a
# technical limit on what this analysis pipeline can process -- so
# .mkv/.ts files in the dataset are handled here without needing to
# re-encode or modify any of the user's original files.
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts"}

_MOVEMENT_MODULES = {
    "footwork": footwork_analysis,
    "smash": smash_analysis,
    "jump": jump_analysis,
    "lunge": lunge_analysis,
}
_VALID_MOVEMENT_TYPES = set(_MOVEMENT_MODULES.keys())


def _parse_class_folder_name(folder_name: str) -> Optional[Tuple[str, str]]:
    """
    Parse a class-folder name like "Good_Footwork" or "poor_Jump" into
    (quality_label, video_type). Matching is case-insensitive.

    Returns None if the folder name doesn't match the expected
    "<Good|Poor>_<MovementType>" pattern.
    """
    parts = folder_name.split("_", 1)
    if len(parts) != 2:
        return None
    quality_raw, movement_raw = parts
    quality = quality_raw.strip().capitalize()  # "good"/"Good"/"GOOD" -> "Good"
    movement = movement_raw.strip().lower()
    if quality not in ("Good", "Poor"):
        return None
    if movement not in _VALID_MOVEMENT_TYPES:
        return None
    return quality, movement


def _discover_dataset(raw_videos_dir: str) -> Dict[str, List[str]]:
    """Return {class_folder_name: [absolute video file paths]} for every
    class folder under dataset/raw_videos/, skipping non-video files."""
    dataset: Dict[str, List[str]] = {}
    if not os.path.isdir(raw_videos_dir):
        return dataset
    for entry in sorted(os.listdir(raw_videos_dir)):
        folder_path = os.path.join(raw_videos_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        videos = []
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                videos.append(os.path.join(folder_path, fname))
        if videos:
            dataset[entry] = videos
    return dataset


# ---------------------------------------------------------------------
# Part 1-3: dataset structure / class balance / issue verification
# (no pose estimation runs here -- this is a fast, file-level check)
# ---------------------------------------------------------------------

def verify_dataset(raw_videos_dir: str = RAW_VIDEOS_DIR) -> Dict[str, object]:
    """
    Inspect the dataset folder structure and report class balance,
    naming issues, non-web-whitelisted formats, and duplicate files
    (by content hash, not just filename).
    """
    dataset = _discover_dataset(raw_videos_dir)

    report: Dict[str, object] = {
        "class_folders": {},
        "total_videos": 0,
        "naming_issues": [],
        "unsupported_files": [],
        "unparseable_folders": [],
        "class_balance": {},
        "duplicate_files": {},
    }

    seen_hashes: Dict[str, List[str]] = {}

    for folder_name, video_paths in dataset.items():
        parsed = _parse_class_folder_name(folder_name)
        if parsed is None:
            report["unparseable_folders"].append(folder_name)
            continue
        quality, movement = parsed

        expected_name = f"{quality}_{movement.capitalize()}"
        if folder_name != expected_name:
            report["naming_issues"].append(
                f"Folder '{folder_name}' should be named '{expected_name}' for "
                "consistent casing (handled case-insensitively here, but "
                "renaming on disk is recommended for clarity)."
            )

        report["class_folders"][folder_name] = {
            "quality_label": quality,
            "video_type": movement,
            "video_count": len(video_paths),
        }
        report["total_videos"] += len(video_paths)

        for path in video_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in {".mp4", ".avi", ".mov"}:
                report["unsupported_files"].append(
                    {
                        "file": os.path.relpath(path, raw_videos_dir),
                        "reason": (
                            f"{ext} is not in the live app's upload whitelist "
                            "(.mp4/.avi/.mov) -- fine for this offline "
                            "extraction script, which reads via OpenCV directly."
                        ),
                    }
                )
            with open(path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            seen_hashes.setdefault(digest, []).append(os.path.relpath(path, raw_videos_dir))

    report["duplicate_files"] = {h: files for h, files in seen_hashes.items() if len(files) > 1}

    balance: Dict[str, Dict[str, int]] = {}
    for info in report["class_folders"].values():
        movement, quality = info["video_type"], info["quality_label"]
        balance.setdefault(movement, {"Good": 0, "Poor": 0})
        balance[movement][quality] += info["video_count"]
    report["class_balance"] = balance

    return report


# ---------------------------------------------------------------------
# Part 4-5: feature extraction (reuses the live pipeline's pure
# computation modules -- pose estimation -> angles -> movement
# analysis -> flat feature row)
# ---------------------------------------------------------------------

def _extract_features_for_video(video_path: str, video_type: str) -> Optional[Dict[str, object]]:
    """
    Run one video through the same pipeline stages as
    services/analysis_service.py's process_video(): read metadata,
    sample frames, run pose detection + joint-angle calculation per
    frame, dispatch to the matching movement-analysis module, and build
    the flat feature row.

    Returns None if no frames could be read, or no person was detected
    in any sampled frame (the caller records this as a failure/skip).
    """
    from modules.pose_estimation import PoseEstimator  # lazy: see note at top of file

    info = video_utils.get_video_information(video_path)
    sampled = video_utils.sample_frames(video_path, sample_rate=FRAME_SAMPLE_RATE)
    if not sampled:
        # The container opened and metadata was readable (that already
        # succeeded, above), but not a single frame could actually be
        # decoded -- this is the signature of a codec OpenCV's build
        # can't handle (e.g. AV1-encoded .mkv files with a pip-installed
        # opencv-python, which commonly lacks AV1 decode support), not a
        # "no person visible" situation.
        raise ValueError(
            "No frames could be decoded from this file (0 frames read despite "
            "readable metadata) -- likely an unsupported video codec in this "
            "environment's OpenCV build, e.g. AV1."
        )

    frame_records = []
    person_detected_count = 0

    with PoseEstimator() as estimator:
        for frame_index, frame in sampled:
            results = estimator.detect(frame)
            landmarks = estimator.get_landmarks(results)
            if landmarks is None:
                continue
            person_detected_count += 1
            angles = angle_calculator.calculate_all_angles(landmarks)
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "timestamp": round(frame_index / info["fps"], 3) if info["fps"] else 0.0,
                    "landmarks": {lm["name"]: lm for lm in landmarks},
                    "angles": angles,
                }
            )

    if not frame_records:
        return None

    angle_summary = angle_calculator.aggregate_angles([f["angles"] for f in frame_records])
    movement_module = _MOVEMENT_MODULES.get(video_type)
    movement_metrics = movement_module.analyze(frame_records, info["fps"]) if movement_module else {}

    video_info_for_features = {
        **info,
        "sampled_frame_count": len(sampled),
        "person_detected_frame_count": person_detected_count,
    }
    # video_id/user_id don't apply to an offline dataset video (it was
    # never uploaded through the app) -- 0 is used as a clear sentinel.
    # Phase 7B's training script already excludes these identifying
    # columns from the model's actual feature matrix.
    return feature_builder.build_feature_row(
        0, 0, video_type, video_info_for_features, angle_summary, movement_metrics
    )


def _extract_features_worker(video_path: str, video_type: str, result_path: str) -> None:
    """
    Runs _extract_features_for_video() in an isolated subprocess (see
    extract_dataset() below) and writes the result to a temp JSON file.

    Why a subprocess per video: some dataset videos are long/high-
    resolution enough that holding every sampled frame in memory at
    once (utils/video_utils.py's sample_frames(), unchanged) can use
    several GB for a single video. Running each video in its own
    process guarantees that memory is fully released the moment that
    video finishes -- and if one unusually large video does exhaust
    memory, only that one subprocess is killed; the batch continues
    with the next video instead of the whole run crashing.
    """
    try:
        features = _extract_features_for_video(video_path, video_type)
        with open(result_path, "w") as f:
            json.dump({"ok": True, "features": features}, f)
    except Exception as exc:
        with open(result_path, "w") as f:
            json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, f)


def _extract_features_isolated(
    video_path: str, video_type: str, timeout_seconds: int = 300
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """
    Run _extract_features_for_video() for one video in a separate
    process (see _extract_features_worker() above for why).

    Returns (features, error_reason) -- exactly one of the two is set.
    error_reason covers: the subprocess timing out, being killed (e.g.
    by the OS's out-of-memory killer -- a negative exitcode), or a
    normal Python exception raised inside it.
    """
    import multiprocessing
    import tempfile

    try:
        ctx = multiprocessing.get_context("fork")  # Linux/Mac: efficient, avoids re-importing everything
    except ValueError:
        ctx = multiprocessing.get_context()  # Windows: falls back to the platform default ('spawn')

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        result_path = tmp.name

    process = ctx.Process(
        target=_extract_features_worker, args=(video_path, video_type, result_path)
    )
    process.start()
    process.join(timeout_seconds)

    try:
        if process.is_alive():
            process.terminate()
            process.join(5)
            return None, f"Timed out after {timeout_seconds}s (video may be unusually long)"

        if process.exitcode != 0:
            reason = (
                "Process was killed (likely out of memory -- this video may be too "
                "long/high-resolution to hold all sampled frames in memory at once)"
                if process.exitcode is not None and process.exitcode < 0
                else f"Subprocess exited with code {process.exitcode}"
            )
            return None, reason

        if not os.path.isfile(result_path) or os.path.getsize(result_path) == 0:
            return None, "Subprocess produced no result (crashed before writing output)"

        with open(result_path) as f:
            payload = json.load(f)
        if not payload.get("ok"):
            return None, payload.get("error", "Unknown error in subprocess")
        return payload.get("features"), None
    finally:
        if os.path.isfile(result_path):
            os.remove(result_path)


def extract_dataset(
    raw_videos_dir: str = RAW_VIDEOS_DIR, output_csv: str = OUTPUT_CSV
) -> Dict[str, object]:
    """
    Run every video in the dataset through the feature-extraction
    pipeline (each in its own subprocess, see _extract_features_isolated()
    above) and write the results to a CSV ready for Phase 7B's training
    script.
    """
    dataset = _discover_dataset(raw_videos_dir)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    total_videos = sum(len(v) for v in dataset.values())
    processed = 0
    start = time.time()

    for folder_name, video_paths in dataset.items():
        parsed = _parse_class_folder_name(folder_name)
        if parsed is None:
            continue
        quality, movement = parsed

        for path in video_paths:
            processed += 1
            rel_path = os.path.relpath(path, raw_videos_dir)
            print(f"[{processed}/{total_videos}] Processing {rel_path} ...", flush=True)
            features, error = _extract_features_isolated(path, movement)
            if error is not None:
                failures.append({"file": rel_path, "reason": error})
                continue
            if features is None:
                failures.append(
                    {"file": rel_path, "reason": "No person detected in any sampled frame"}
                )
                continue
            features["source_file"] = rel_path
            features["class_folder"] = folder_name
            features["quality_label"] = quality
            rows.append(features)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df.to_csv(output_csv, index=False)

    return {
        "total_videos": total_videos,
        "successful": len(rows),
        "failed": len(failures),
        "failures": failures,
        "output_csv": output_csv,
        "columns": list(df.columns) if not df.empty else [],
        "elapsed_seconds": round(time.time() - start, 1),
    }


def main():
    print("=" * 70)
    print("Phase 7A -- Dataset verification")
    print("=" * 70)
    verification = verify_dataset()
    os.makedirs(os.path.dirname(VERIFICATION_REPORT_PATH), exist_ok=True)
    with open(VERIFICATION_REPORT_PATH, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"Verification report written to {VERIFICATION_REPORT_PATH}")
    print(f"Total videos found: {verification['total_videos']}")
    for folder, info in verification["class_folders"].items():
        print(f"  {folder}: {info['video_count']} videos ({info['quality_label']} {info['video_type']})")
    if verification["naming_issues"]:
        print("\nNaming issues:")
        for issue in verification["naming_issues"]:
            print(f"  - {issue}")
    if verification["duplicate_files"]:
        print("\nDuplicate files found:")
        for files in verification["duplicate_files"].values():
            print(f"  - {files}")

    print("\n" + "=" * 70)
    print("Phase 7A -- Feature extraction")
    print("=" * 70)
    result = extract_dataset()
    print(f"\nProcessed {result['total_videos']} videos in {result['elapsed_seconds']}s")
    print(f"  Succeeded: {result['successful']}")
    print(f"  Failed:    {result['failed']}")
    if result["failures"]:
        print("\nFailures:")
        for fail in result["failures"]:
            print(f"  - {fail['file']}: {fail['reason']}")
    print(f"\nFeature CSV written to: {result['output_csv']}")
    print(f"Columns ({len(result['columns'])}): {result['columns']}")


if __name__ == "__main__":
    main()
