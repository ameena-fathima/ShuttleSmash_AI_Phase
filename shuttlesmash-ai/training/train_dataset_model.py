"""
Phase 7B - Train Random Forest model using the EXISTING train/test split.

IMPORTANT:
- Uses train_features.csv exactly as the training set.
- Uses test_features.csv exactly as the testing set.
- Does NOT create a new split.
- Does NOT reprocess videos.
- Does NOT modify the original videos.
"""

import json
import os
import sys
from datetime import datetime, timezone

# Allow imports from project root if needed later
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
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FEATURES_CSV = os.path.join(
    BASE_DIR,
    "dataset",
    "extracted_features",
    "train_features.csv",
)

TEST_FEATURES_CSV = os.path.join(
    BASE_DIR,
    "dataset",
    "extracted_features",
    "test_features.csv",
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_VERSION = "2.1"
VALID_LABELS = ["Good", "Poor"]


# ============================================================
# COLUMNS THAT MUST NOT BE USED AS ML FEATURES
# ============================================================

NON_FEATURE_COLUMNS = {
    "video_id",
    "user_id",
    "video_type",
    "feature_schema_version",
    "resolution",
    "source_file",
    "class_folder",
    "quality_label",
    "split",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    if not os.path.isfile(TRAIN_FEATURES_CSV):
        raise FileNotFoundError(
            f"Training CSV not found:\n{TRAIN_FEATURES_CSV}"
        )

    if not os.path.isfile(TEST_FEATURES_CSV):
        raise FileNotFoundError(
            f"Testing CSV not found:\n{TEST_FEATURES_CSV}"
        )

    train_df = pd.read_csv(TRAIN_FEATURES_CSV)
    test_df = pd.read_csv(TEST_FEATURES_CSV)

    if "quality_label" not in train_df.columns:
        raise ValueError("Training CSV does not contain quality_label.")

    if "quality_label" not in test_df.columns:
        raise ValueError("Testing CSV does not contain quality_label.")

    train_df = train_df[
        train_df["quality_label"].isin(VALID_LABELS)
    ].copy()

    test_df = test_df[
        test_df["quality_label"].isin(VALID_LABELS)
    ].copy()

    if len(train_df) == 0:
        raise ValueError("No valid training rows found.")

    if len(test_df) == 0:
        raise ValueError("No valid testing rows found.")

    return train_df, test_df


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def train_and_evaluate():

    print("=" * 70)
    print("SHUTTLESMASH AI - PHASE 7B MODEL TRAINING")
    print("=" * 70)

    print("\nLoading existing feature datasets...")

    train_df, test_df = load_dataset()

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    print("\nTraining labels:")
    print(train_df["quality_label"].value_counts().to_dict())

    print("\nTesting labels:")
    print(test_df["quality_label"].value_counts().to_dict())

    # --------------------------------------------------------
    # Verify expected split sizes from our completed extraction
    # --------------------------------------------------------

    if len(train_df) != 401:
        print(
            f"\nWARNING: Expected 401 training rows, "
            f"but found {len(train_df)}."
        )

    if len(test_df) != 106:
        print(
            f"\nWARNING: Expected 106 testing rows, "
            f"but found {len(test_df)}."
        )

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No usable feature columns found.")

    print(f"\nNumber of ML features: {len(feature_columns)}")

    # Make sure train and test contain the same features
    missing_in_test = [
        column
        for column in feature_columns
        if column not in test_df.columns
    ]

    if missing_in_test:
        raise ValueError(
            f"Features missing from test dataset: {missing_in_test}"
        )

    X_train = (
        train_df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    X_test = (
        test_df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y_train = train_df["quality_label"]
    y_test = test_df["quality_label"]

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # IMPORTANT:
    # Test data is transformed using the scaler fitted ONLY on
    # the training data.
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest classifier...")
    print("This uses the existing train/test split.")
    print("No new split is being created.")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    print("Model training complete.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nEvaluating on the existing test dataset...")

    y_pred = model.predict(X_test_scaled)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    conf_matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=VALID_LABELS,
    ).tolist()

    report_text = classification_report(
        y_test,
        y_pred,
        labels=VALID_LABELS,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importances = sorted(
        zip(
            feature_columns,
            model.feature_importances_.tolist(),
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )

    top_features = [
        {
            "feature": name,
            "importance": round(value, 4),
        }
        for name, value in importances[:15]
    ]

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODELS_DIR,
        "model.joblib",
    )

    scaler_path = os.path.join(
        MODELS_DIR,
        "scaler.joblib",
    )

    metadata_path = os.path.join(
        MODELS_DIR,
        "model_metadata.json",
    )

    evaluation_path = os.path.join(
        REPORTS_DIR,
        "model_evaluation.json",
    )

    classification_path = os.path.join(
        REPORTS_DIR,
        "classification_report.txt",
    )

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier",
        "training_dataset": "train_features.csv",
        "testing_dataset": "test_features.csv",
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "training_label_counts": {
            str(k): int(v)
            for k, v in train_df["quality_label"]
            .value_counts()
            .to_dict()
            .items()
        },
        "testing_label_counts": {
            str(k): int(v)
            for k, v in test_df["quality_label"]
            .value_counts()
            .to_dict()
            .items()
        },
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "labels": VALID_LABELS,
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "split_method": "pre-existing train_features.csv and test_features.csv",
        "new_split_created": False,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    # --------------------------------------------------------
    # Evaluation report
    # --------------------------------------------------------

    evaluation = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "confusion_matrix": conf_matrix,
        "confusion_matrix_labels": VALID_LABELS,
        "top_feature_importances": top_features,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
    }

    with open(
        evaluation_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(evaluation, f, indent=2)

    with open(
        classification_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_text)

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nTraining rows : {len(train_df)}")
    print(f"Testing rows  : {len(test_df)}")
    print(f"ML features   : {len(feature_columns)}")

    print("\nRESULTS")
    print("-" * 40)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion Matrix")
    print(f"Labels: {VALID_LABELS}")
    print(conf_matrix)

    print("\nTop features:")
    for item in top_features[:10]:
        print(
            f"  {item['feature']}: "
            f"{item['importance']:.4f}"
        )

    print("\nSaved files:")
    print(f"Model      -> {model_path}")
    print(f"Scaler     -> {scaler_path}")
    print(f"Metadata   -> {metadata_path}")
    print(f"Evaluation -> {evaluation_path}")
    print(f"Report     -> {classification_path}")

    print("\n" + "=" * 70)

    return metadata, evaluation


if __name__ == "__main__":
    train_and_evaluate()