# models/

Holds the trained ML artifacts, created by running:

    python training/train_model.py

| file                    | contents                                                  |
|--------------------------|-----------------------------------------------------------|
| `model.joblib`           | the trained `RandomForestClassifier` (scikit-learn)        |
| `scaler.joblib`          | the fitted `StandardScaler` used to normalize features before prediction |
| `model_metadata.json`    | model version, training timestamp, accuracy, and the exact list/order of feature columns the model expects |

## How the app uses this folder

`modules/ml_predictor.py` checks for `model.joblib` + `scaler.joblib` on
first use. If both exist, it loads them once and reuses them for every
prediction (same "load once, reuse many times" pattern as
`modules/pose_estimation.py`'s `PoseEstimator`). If either file is
missing, or fails to load for any reason, the app **automatically falls
back** to the existing Phase 4 rule-based injury-risk detection --
nothing else in the app breaks or changes behavior.

## Replacing the model

To deploy a newly retrained model, just overwrite these three files
(e.g. by running `training/train_model.py` again, or by copying in
files trained elsewhere with the same feature layout). The app picks up
a new model automatically the next time it's restarted -- no code
changes needed.

No model files are committed to version control from this folder (see
the project's `.gitignore`) -- each installation trains its own model
from its own analyzed videos.
