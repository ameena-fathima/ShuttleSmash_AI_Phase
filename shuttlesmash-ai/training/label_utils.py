"""
training/label_utils.py
--------------------------
Simple labeling workflow: associates each analyzed video (identified by
analysis_id) with a risk label for supervised training.

Two ways to populate dataset/labels/labels.csv:
  1. Auto-suggested (this file): run suggest_labels_from_rules(), which
     seeds one row per analyzed video with a STARTING-POINT label
     derived from the existing Phase 4 rule-based `overall_score`. This
     is only a convenience seed, not ground truth -- the whole point of
     training an ML model is for it to eventually learn patterns beyond
     the fixed rule thresholds, so review and correct these by hand
     (ideally by watching the actual footage) before training.
  2. Manual: open dataset/labels/labels.csv afterward and edit the
     `label` column directly.

Run standalone:
    python training/label_utils.py
"""

import csv
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The fixed label set this whole ML pipeline is built around -- used
# here, in training/train_model.py, and in modules/ml_predictor.py.
VALID_LABELS = ["Normal", "Low Risk", "Medium Risk", "High Risk"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABELS_PATH = os.path.join(BASE_DIR, "dataset", "labels", "labels.csv")


def _score_to_label(overall_score: Optional[float]) -> str:
    """
    Map the existing Phase 4 rule-based overall_score (0-100) onto the
    4-class label set, as a starting-point suggestion only.
    """
    if overall_score is None:
        return ""
    if overall_score >= 85:
        return "Normal"
    if overall_score >= 70:
        return "Low Risk"
    if overall_score >= 50:
        return "Medium Risk"
    return "High Risk"


def suggest_labels_from_rules(output_path: str = DEFAULT_LABELS_PATH) -> int:
    """
    Seed dataset/labels/labels.csv with one row per analyzed video,
    using the existing rule-based overall_score as a starting-point
    label suggestion.

    Safe to re-run: it overwrites the file from scratch based on
    whatever's in the database right now. If you've already hand-edited
    labels and want to keep them, back up the file first or merge
    manually -- this function intentionally does not try to guess which
    edits were "yours" vs. auto-generated.

    Returns:
        The number of rows written.
    """
    from app import create_app
    from database.db_manager import get_db

    app = create_app()
    with app.app_context():
        db = get_db()
        rows = db.execute(
            "SELECT id AS analysis_id, video_id, overall_score FROM analyses ORDER BY id"
        ).fetchall()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["analysis_id", "video_id", "suggested_label", "label"])
        for row in rows:
            suggested = _score_to_label(row["overall_score"])
            # `label` starts equal to `suggested_label` -- edit this
            # column by hand before training if you disagree.
            writer.writerow([row["analysis_id"], row["video_id"], suggested, suggested])

    print(
        f"Wrote {len(rows)} label row(s) to {output_path}. "
        "Review/correct the 'label' column before training."
    )
    return len(rows)


def load_labels(path: str = DEFAULT_LABELS_PATH) -> Dict[int, str]:
    """
    Load labels.csv into a {analysis_id: label} dict, used by
    training/train_model.py to attach labels to feature rows.

    Rows with an empty or invalid label are silently skipped, so a
    partially-labeled dataset can still be used to train on whatever
    rows do have a valid label.
    """
    labels: Dict[int, str] = {}
    if not os.path.isfile(path):
        return labels

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label") or "").strip()
            if label in VALID_LABELS:
                try:
                    labels[int(row["analysis_id"])] = label
                except (KeyError, ValueError):
                    continue  # malformed row -- skip rather than crash

    return labels


if __name__ == "__main__":
    suggest_labels_from_rules()
