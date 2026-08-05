"""
modules/ml_predictor.py
--------------------------
Loads the trained Random Forest model (if one exists) and predicts a
badminton injury-risk label from the same flat feature dict Phase 3's
modules/feature_builder.py already builds for every analyzed video.

If no trained model is present yet (models/model.joblib missing, or it
fails to load for any reason), every function here degrades gracefully:
predict_risk_level() returns None, and services/analysis_service.py
falls back to the existing Phase 4 rule-based injury_risk.py detection,
exactly as it did before Phase 5. No existing behavior changes until a
model is actually trained and saved -- see training/train_model.py.
"""

import json
import os
from typing import Dict, List, Optional

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Cached in-process so the model file is only read from disk once, not
# on every single video processed -- the same "load once, reuse many
# times" principle modules/pose_estimation.py's PoseEstimator uses.
_cached_model = None
_cached_scaler = None
_cached_metadata: Optional[Dict] = None
_load_attempted = False


def _load_model_files() -> None:
    """Load model/scaler/metadata from disk once and cache them in
    module-level globals. Safe to call repeatedly -- only touches disk
    the first time it's called."""
    global _cached_model, _cached_scaler, _cached_metadata, _load_attempted

    if _load_attempted:
        return
    _load_attempted = True

    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(SCALER_PATH)):
        return  # no trained model yet -- callers fall back to rule-based

    try:
        import joblib

        _cached_model = joblib.load(MODEL_PATH)
        _cached_scaler = joblib.load(SCALER_PATH)
        if os.path.isfile(METADATA_PATH):
            with open(METADATA_PATH) as f:
                _cached_metadata = json.load(f)
    except Exception:
        # Any load failure (corrupted file, version mismatch, missing
        # scikit-learn, etc.) is treated the same as "no model" -- the
        # app must keep working via the rule-based fallback either way.
        _cached_model = None
        _cached_scaler = None
        _cached_metadata = None


def reset_cache() -> None:
    """
    Force the next call to reload model files from disk (e.g. after
    training a new model while the app is still running). Not called
    automatically anywhere -- provided for a future "reload model"
    admin action or for tests.
    """
    global _cached_model, _cached_scaler, _cached_metadata, _load_attempted
    _cached_model = None
    _cached_scaler = None
    _cached_metadata = None
    _load_attempted = False


def is_model_available() -> bool:
    """True if a trained model was found and successfully loaded."""
    _load_model_files()
    return _cached_model is not None and _cached_scaler is not None


def get_model_status() -> Dict[str, object]:
    """
    Return the small set of facts the dashboard shows about the ML
    model: status, version, training date, accuracy.

    Always returns a dict, even when no model has been trained yet, so
    the dashboard template never has to special-case a missing model.
    """
    _load_model_files()
    if not is_model_available() or _cached_metadata is None:
        return {
            "status": "Not Trained",
            "model_version": None,
            "trained_at": None,
            "accuracy": None,
        }
    return {
        "status": "Trained",
        "model_version": _cached_metadata.get("model_version"),
        "trained_at": _cached_metadata.get("trained_at"),
        "accuracy": _cached_metadata.get("accuracy"),
    }


def _to_number(value: object) -> float:
    """Coerce a feature value to a float for the model, treating
    missing/non-numeric values as 0 -- matches training/train_model.py's
    `fillna(0)` handling of the same feature columns, so predictions use
    the same convention the model was trained under."""
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def predict_risk_level(features: Dict[str, object]) -> Optional[Dict[str, object]]:
    """
    Predict an overall risk label for one analyzed video from its flat
    feature dict (the same shape modules/feature_builder.py produces
    and services/analysis_service.py saves as `analyses.features_json`).

    Args:
        features: a feature dict, e.g. the return value of
            modules/feature_builder.py's build_feature_row(), or the
            same dict reloaded from a saved analysis's features_json.

    Returns:
        None if no trained model is available or prediction fails for
        any reason -- callers MUST fall back to the rule-based
        modules/injury_risk.py detection in that case (see
        services/analysis_service.py), per Phase 5's requirement to
        never break existing functionality without a trained model.

        Otherwise:
            {
                "label": "Low Risk", "confidence": 82.4,
                "source": "Machine Learning", "model_version": "1.0",
            }
    """
    _load_model_files()
    if not is_model_available():
        return None

    feature_columns: List[str] = (_cached_metadata or {}).get("feature_columns", [])
    if not feature_columns:
        return None

    # Built as a DataFrame with the same column names training used
    # (see training/train_model.py) rather than a plain list -- this
    # matches what the scaler/model were fit on and avoids a harmless
    # but noisy "X does not have valid feature names" scikit-learn
    # warning that a plain list input would otherwise trigger.
    import pandas as pd

    row = pd.DataFrame(
        [[_to_number(features.get(col)) for col in feature_columns]],
        columns=feature_columns,
    )

    try:
        scaled = _cached_scaler.transform(row)
        predicted_label = _cached_model.predict(scaled)[0]
        probabilities = _cached_model.predict_proba(scaled)[0]
        confidence = round(float(max(probabilities)) * 100, 1)
    except Exception:
        # A shape/feature-layout mismatch (e.g. the model was trained on
        # an older feature_builder.py version) is treated as "no
        # prediction" rather than crashing the whole pipeline.
        return None

    return {
        "label": str(predicted_label),
        "confidence": confidence,
        "source": "Machine Learning",
        "model_version": (_cached_metadata or {}).get("model_version"),
    }
