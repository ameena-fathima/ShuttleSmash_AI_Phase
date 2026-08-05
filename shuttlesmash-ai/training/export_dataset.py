"""
training/export_dataset.py
-----------------------------
Exports every analyzed video's stored ML feature row into a single CSV
file for training.

IMPORTANT: this does NOT recompute any features. It only reads the flat
feature dict Phase 3's modules/feature_builder.py already built and
saved into the `analyses.features_json` column when the video was
processed (see services/analysis_service.py). This keeps feature
calculation logic in exactly one place, per the project's "do not
duplicate calculations" requirement for Phase 5.

Run standalone:
    python training/export_dataset.py

Or import and call from elsewhere:
    from training.export_dataset import export_features_to_csv
    export_features_to_csv()
"""

import json
import os
import sys

# Allow running this file directly (`python training/export_dataset.py`)
# by putting the project root on sys.path, the same way a package import
# would resolve it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "extracted_features", "features.csv")


def export_features_to_csv(output_path: str = DEFAULT_OUTPUT_PATH) -> int:
    """
    Read every row of the `analyses` table that has a saved
    `features_json`, flatten it, and write the result to a CSV file.

    Args:
        output_path: where to write the CSV (default: see above).

    Returns:
        The number of rows written (0 if no analyzed videos exist yet).
    """
    # Reuse the existing Flask app factory + db_manager -- this is the
    # same database connection code every other part of the app uses,
    # not a second, duplicate way of talking to SQLite.
    from app import create_app
    from database.db_manager import get_db

    app = create_app()
    with app.app_context():
        db = get_db()
        rows = db.execute(
            """
            SELECT id AS analysis_id, video_id, user_id, features_json,
                   overall_score, analysis_date
            FROM analyses
            WHERE features_json IS NOT NULL
            ORDER BY id
            """
        ).fetchall()

    records = []
    for row in rows:
        features = json.loads(row["features_json"])
        # Add a few identifying/label-adjacent columns that live outside
        # the feature_builder dict itself, so the exported CSV can be
        # joined against dataset/labels/labels.csv by analysis_id.
        features["analysis_id"] = row["analysis_id"]
        features["overall_score"] = row["overall_score"]
        features["analysis_date"] = row["analysis_date"]
        records.append(features)

    if not records:
        print(
            "No analyzed videos found yet -- upload and analyze a few "
            "videos through the app first, then re-run this script."
        )
        return 0

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} row(s) and {len(df.columns)} columns to {output_path}")
    return len(df)


if __name__ == "__main__":
    export_features_to_csv()
