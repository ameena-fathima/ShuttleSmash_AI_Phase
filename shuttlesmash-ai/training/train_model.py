"""
training/train_model.py
--------------------------
Trains a Random Forest classifier to predict a badminton injury-risk
label from the same joint-angle / movement-metric features already
computed by Phase 3 (modules/feature_builder.py) and exported to CSV by
training/export_dataset.py.

Full workflow:
    python training/export_dataset.py     # refresh dataset/extracted_features/features.csv
    python training/label_utils.py        # seed dataset/labels/labels.csv (then edit by hand)
    python training/train_model.py        # train + evaluate + save the model

Outputs:
    models/model.joblib               the trained RandomForestClassifier
    models/scaler.joblib              the fitted StandardScaler
    models/model_metadata.json        version / training date / accuracy / feature column list
    dataset/train/train_split.csv     the exact rows used for training (reproducibility)
    dataset/test/test_split.csv       the exact rows used for evaluation
    reports/model_evaluation.json     accuracy / precision / recall / f1 / confusion matrix
    reports/classification_report.txt   scikit-learn's full per-class report

This script can be run completely independently of the Flask app (it
only needs the SQLite database file to already exist with some analyzed
videos in it) -- it does not need the app server to be running.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.label_utils import VALID_LABELS, load_labels

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(BASE_DIR, "dataset", "extracted_features", "features.csv")
LABELS_CSV = os.path.join(BASE_DIR, "dataset", "labels", "labels.csv")
TRAIN_SPLIT_CSV = os.path.join(BASE_DIR, "dataset", "train", "train_split.csv")
TEST_SPLIT_CSV = os.path.join(BASE_DIR, "dataset", "test", "test_split.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_VERSION = "1.0"

# Columns present in features.csv that identify a row or aren't numeric
# model inputs -- excluded from the feature matrix. Everything else
# (all the joint-angle stats and movement metrics) is used as-is.
NON_FEATURE_COLUMNS = {
    "video_id",
    "user_id",
    "video_type",
    "feature_schema_version",
    "resolution",
    "analysis_id",
    "analysis_date",
    "overall_score",
}


def _load_training_data() -> pd.DataFrame:
    """Load features.csv, attach labels.csv, and drop unlabeled rows."""
    if not os.path.isfile(FEATURES_CSV):
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found -- run `python training/export_dataset.py` first."
        )

    df = pd.read_csv(FEATURES_CSV)
    labels = load_labels(LABELS_CSV)
    if not labels:
        raise ValueError(
            f"No labels found in {LABELS_CSV} -- run `python training/label_utils.py`, "
            "then fill in the 'label' column, before training."
        )

    df["label"] = df["analysis_id"].map(labels)
    df = df.dropna(subset=["label"])

    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} labeled row(s) found -- at least 10 are needed for a "
            "meaningful train/test split. Analyze more videos and label them first."
        )

    return df


def train_and_evaluate() -> tuple:
    """
    Run the full train + evaluate + save pipeline.

    Returns:
        (metadata, evaluation) -- the two dicts also written to disk as
        models/model_metadata.json and reports/model_evaluation.json.
    """
    df = _load_training_data()

    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "label"]
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["label"]

    # Stratify keeps the class proportions similar in train vs. test,
    # but only works if every class has at least 2 examples -- fall back
    # to a plain random split on very small/imbalanced datasets.
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Save the exact split used, for reproducibility (Part 1's train/
    # test folders).
    os.makedirs(os.path.dirname(TRAIN_SPLIT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_SPLIT_CSV), exist_ok=True)
    X_train.assign(label=y_train).to_csv(TRAIN_SPLIT_CSV, index=False)
    X_test.assign(label=y_test).to_csv(TEST_SPLIT_CSV, index=False)

    # Scale features before training -- Random Forest doesn't strictly
    # require this, but keeping a scaler in the pipeline means the same
    # preprocessing step is available if the model is ever swapped for
    # one that does need scaled inputs (e.g. logistic regression, SVM).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=VALID_LABELS).tolist()
    report_text = classification_report(y_test, y_pred, labels=VALID_LABELS, zero_division=0)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier",
        "accuracy": round(float(accuracy), 4),
        "feature_columns": feature_columns,
        "labels": VALID_LABELS,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    evaluation = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "confusion_matrix": conf_matrix,
        "confusion_matrix_labels": VALID_LABELS,
    }
    with open(os.path.join(REPORTS_DIR, "model_evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)

    print(
        f"Model trained on {len(X_train)} rows, evaluated on {len(X_test)} rows.\n"
        f"  Accuracy:  {accuracy:.3f}\n"
        f"  Precision: {precision:.3f} (weighted)\n"
        f"  Recall:    {recall:.3f} (weighted)\n"
        f"  F1-score:  {f1:.3f} (weighted)\n"
        f"Saved model     -> {MODELS_DIR}/model.joblib\n"
        f"Saved scaler    -> {MODELS_DIR}/scaler.joblib\n"
        f"Saved metadata  -> {MODELS_DIR}/model_metadata.json\n"
        f"Saved evaluation -> {REPORTS_DIR}/model_evaluation.json\n"
        f"Saved report    -> {REPORTS_DIR}/classification_report.txt"
    )
    return metadata, evaluation


if __name__ == "__main__":
    train_and_evaluate()
