"""
modules/feature_builder.py
-----------------------------
Builds the single, flat "feature row" dictionary for one analyzed video
-- the same shape that will later be written straight into a pandas
DataFrame / CSV file for training a Random Forest injury-risk model
(Phase 4+).

Keeping this in one place, with a fixed key layout, means:
  - the `analyses.features_json` column in the database always has a
    consistent shape to query/inspect,
  - a future CSV export can simply loop over saved analyses, call
    `json.loads(row["features_json"])`, and get ML-ready rows with no
    extra transformation logic -- e.g.:

        import json, pandas as pd
        rows = [json.loads(r["features_json"]) for r in all_analyses]
        df = pd.DataFrame(rows)
        df.to_csv("training_data.csv", index=False)

No injury-risk logic or scoring lives here -- this module only flattens
already-computed numbers into one dict.
"""

from typing import Dict, Optional

# Bump this if the feature layout ever changes shape, so old saved rows
# can be told apart from new ones when building a training CSV later.
FEATURE_SCHEMA_VERSION = "1.0"

# Movement-metric keys get prefixed by video type so footwork/smash/jump/
# lunge features never collide by name in the flat row, and so every
# saved row has a predictable, ML-friendly key set (a row for a "smash"
# video simply won't have any footwork_*/jump_*/lunge_* keys set to a
# real value -- Phase 4's CSV export can fill those with NaN).
_CATEGORY_PREFIX = {
    "footwork": "footwork_",
    "smash": "smash_",
    "jump": "jump_",
    "lunge": "lunge_",
}


def build_feature_row(
    video_id: int,
    user_id: int,
    video_type: str,
    video_info: Dict[str, object],
    angle_summary: Dict[str, Dict[str, Optional[float]]],
    movement_metrics: Dict[str, Optional[float]],
) -> Dict[str, object]:
    """
    Flatten everything computed for one video into a single dict with
    flat, consistently-named, ML-friendly keys.

    Args:
        video_id, user_id, video_type: identifying/context info.
        video_info: dict from utils/video_utils.get_video_information(),
            plus "sampled_frame_count" and "person_detected_frame_count".
        angle_summary: output of angle_calculator.aggregate_angles().
        movement_metrics: output of the video-type-specific analysis
            module (footwork_analysis.analyze(), etc.). Pass {} if the
            video type has no dedicated module (e.g. "general").

    Returns:
        A flat dict, e.g.:
            {
                "video_id": 12, "user_id": 3, "video_type": "smash",
                "feature_schema_version": "1.0",
                "fps": 29.97, "duration_seconds": 8.2, ...
                "left_elbow_angle_avg": 142.1, "left_elbow_angle_min": 90.0,
                "left_elbow_angle_max": 178.4, ...
                "smash_max_elbow_extension_angle": 176.2, ...
            }
    """
    row: Dict[str, object] = {
        "video_id": video_id,
        "user_id": user_id,
        "video_type": video_type,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fps": video_info.get("fps"),
        "duration_seconds": video_info.get("duration_seconds"),
        "resolution": video_info.get("resolution"),
        "frame_count": video_info.get("frame_count"),
        "sampled_frame_count": video_info.get("sampled_frame_count"),
        "person_detected_frame_count": video_info.get("person_detected_frame_count"),
    }

    # Flatten joint angle summary: left_elbow_angle -> left_elbow_angle_avg/min/max
    for angle_name, stats in angle_summary.items():
        for stat_name, value in stats.items():
            row[f"{angle_name}_{stat_name}"] = value

    # Flatten movement metrics with a category prefix (see _CATEGORY_PREFIX above).
    prefix = _CATEGORY_PREFIX.get(video_type, "")
    for metric_name, value in movement_metrics.items():
        row[f"{prefix}{metric_name}"] = value

    return row
