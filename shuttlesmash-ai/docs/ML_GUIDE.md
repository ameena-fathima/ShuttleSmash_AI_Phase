# ShuttleSmash AI -- Machine Learning Guide (Phase 5)

This document explains how the ML-based injury-risk classifier fits
into ShuttleSmash AI, how to build a training dataset from analyzed
videos, how to train and evaluate the model, and how to retrain or
replace it later.

## 1. Where ML fits into the existing pipeline

Phases 1-4 already built a complete, working pipeline:

```
Video -> Pose Estimation -> Joint Angles -> Movement Analysis
      -> Rule-Based Injury Risk Detection -> Recommendations -> Results Page
```

Phase 5 does **not** replace any of that. It adds one more, optional
step at the end, reusing everything before it:

```
... -> Rule-Based Injury Risk Detection (unchanged, always runs)
     -> [NEW] ML Model Prediction (only if a model has been trained)
     -> Results Page (shows both, side by side)
```

`modules/ml_predictor.py` checks whether a trained model exists
(`models/model.joblib`). If it does, it predicts an overall risk label
(`Normal` / `Low Risk` / `Medium Risk` / `High Risk`) and the results
page shows it as a labeled "ML Model Prediction" alongside the existing
rule-based risk cards. If no model has been trained yet -- which is the
state of a fresh installation -- `predict_risk_level()` returns `None`
and the app falls back to showing only the rule-based results, exactly
as it did before Phase 5. **Nothing about the existing rule-based
system was changed or removed.**

## 2. Feature extraction (reused, not duplicated)

Every video analyzed through the app already has its joint-angle
statistics and movement metrics flattened into one row by
`modules/feature_builder.py` (built in Phase 3) and saved as JSON in the
`analyses.features_json` database column.

`training/export_dataset.py` reads that column straight out of the
database and writes it to `dataset/extracted_features/features.csv` --
no feature is recalculated. This is intentional: it guarantees the
exact same numbers a player saw on their results page are what the
model is trained on, and it means any future improvement to
`feature_builder.py` automatically flows through to the next dataset
export with zero extra work.

Feature columns include, for both left and right sides:
- `*_elbow_angle_avg/min/max`, `*_shoulder_angle_avg/min/max`,
  `*_hip_angle_avg/min/max`, `*_knee_angle_avg/min/max`,
  `*_ankle_angle_avg/min/max` (degrees, from `modules/angle_calculator.py`)
- Movement metrics specific to the video's type, prefixed accordingly:
  `footwork_*`, `smash_*`, `jump_*`, `lunge_*`
- Frame/quality statistics: `fps`, `duration_seconds`, `frame_count`,
  `sampled_frame_count`, `person_detected_frame_count`

Left/right asymmetry isn't a separate stored column -- it's derived at
training time by comparing the corresponding `left_*` / `right_*`
columns already present (the same approach `modules/injury_risk.py`
uses for its own rule-based asymmetry checks).

## 3. Label pipeline

The model predicts one of four labels:

- `Normal`
- `Low Risk`
- `Medium Risk`
- `High Risk`

`training/label_utils.py` seeds `dataset/labels/labels.csv` with one row
per analyzed video, pre-filled with a **suggested** label derived from
the existing rule-based `overall_score`:

| overall_score | suggested_label |
|----------------|-------------------|
| 85-100         | Normal            |
| 70-84          | Low Risk           |
| 50-69          | Medium Risk        |
| 0-49           | High Risk          |

This is only a convenience starting point. Before training, open the
CSV and correct the `label` column by hand (ideally after watching the
actual footage/skeleton preview) -- the entire value of training a
separate ML model is for it to eventually generalize beyond these fixed
rule thresholds, which it can only do if it learns from real human
judgement, not from the rules it's meant to complement.

## 4. Model architecture

- **Algorithm:** `RandomForestClassifier` (scikit-learn), 200 trees.
  Chosen because it's robust to unscaled/noisy features, handles a
  mixed bag of angle and movement-metric columns well without much
  tuning, and -- important for a viva -- its `feature_importances_`
  can be inspected to explain *which* joint angles and movement metrics
  the model actually relies on, keeping it reasonably explainable
  compared to a black-box neural network.
- **Preprocessing:** a `StandardScaler` is fit on the training split and
  applied to both train and test data before fitting the model. Random
  Forests don't strictly require feature scaling, but keeping this step
  in the pipeline means the same preprocessing code works unchanged if
  the model is ever swapped for one that does need it (e.g. logistic
  regression, SVM, a neural network).
- **Split:** 80% train / 20% test, stratified by label where possible,
  `random_state=42` for reproducibility.
- **Inputs:** every numeric column in `features.csv` except identifying
  columns (`video_id`, `user_id`, `analysis_id`, etc. -- see
  `NON_FEATURE_COLUMNS` in `training/train_model.py`). Missing values
  are filled with 0 (matches how `modules/ml_predictor.py` treats
  missing features at prediction time, so training and inference agree).
- **Output:** one of the 4 risk labels, plus a confidence score (the
  model's predicted probability for that label).

## 5. How to build a dataset and train a model

```bash
# 1. Analyze several videos through the normal app flow first
#    (register/login -> /upload -> pick a video + type -> submit)

# 2. Export every analyzed video's stored features to CSV
python training/export_dataset.py

# 3. Seed labels.csv with rule-based suggestions, then EDIT the
#    'label' column by hand before continuing
python training/label_utils.py
#    -> open dataset/labels/labels.csv and review/correct each row

# 4. Train, evaluate, and save the model
python training/train_model.py
```

`train_model.py` needs at least 10 labeled rows to run (a meaningful
train/test split isn't possible with fewer). More labeled, correctly
reviewed examples -- ideally dozens to hundreds, covering a range of
techniques and risk levels -- will produce a meaningfully better model
than a handful of videos.

## 6. Evaluation outputs

Every training run writes:

- `models/model_metadata.json` -- version, training timestamp, accuracy,
  and the exact list of feature columns the model expects (used by
  `modules/ml_predictor.py` at prediction time).
- `reports/model_evaluation.json` -- accuracy, weighted precision/
  recall/F1, and the confusion matrix.
- `reports/classification_report.txt` -- scikit-learn's full per-class
  precision/recall/F1 breakdown.
- `dataset/train/train_split.csv` / `dataset/test/test_split.csv` -- the
  exact rows used, for reproducibility.

The dashboard (Part 7) shows a live summary of `model_metadata.json`:
model status, version, training date, and accuracy.

## 7. How to retrain

Just re-run `python training/train_model.py` any time you've analyzed
and labeled more videos. It overwrites `models/model.joblib`,
`models/scaler.joblib`, and `models/model_metadata.json` in place --
restart the Flask app afterward (or wait for the next process restart)
to pick up the new model, since `modules/ml_predictor.py` caches the
loaded model in memory for performance.

## 8. How to replace the model

To use a model trained elsewhere (a different machine, a different
algorithm, more data), just replace these three files with matching
ones:

- `models/model.joblib` -- any scikit-learn-compatible classifier
  object with `.predict()` and `.predict_proba()`.
- `models/scaler.joblib` -- any object with a `.transform()` method
  matching what the model was trained on (a no-op passthrough scaler is
  fine if the replacement model doesn't need scaling).
- `models/model_metadata.json` -- must include a `feature_columns` list
  matching, in order, what the model expects as input, plus whatever
  `model_version` / `trained_at` / `accuracy` values you want the
  dashboard to display.

No code changes are required -- `modules/ml_predictor.py` only depends
on this fixed file/field contract.

## 9. Folder structure (Phase 5 additions)

```
shuttlesmash-ai/
├── dataset/
│   ├── README.md
│   ├── raw_videos/            optional holding area for footage
│   ├── extracted_features/    features.csv (from export_dataset.py)
│   ├── labels/                labels.csv (from label_utils.py)
│   ├── train/                 train_split.csv (from train_model.py)
│   ├── validation/            reserved for a future k-fold setup
│   └── test/                  test_split.csv (from train_model.py)
├── models/
│   ├── model.joblib
│   ├── scaler.joblib
│   └── model_metadata.json
├── training/
│   ├── export_dataset.py      Part 2
│   ├── label_utils.py         Part 3
│   └── train_model.py         Part 4 + 5
└── modules/
    └── ml_predictor.py        Part 6 (inference + fallback)
```

See `docs/architecture.md` for the complete project-wide architecture
across all phases.

---

## 10. Phase 7: training on your own labeled video dataset

Phases 1-6 above describe training on videos analyzed through the live
app, using rule-suggested labels you review by hand. Phase 7 adds a
**second, independent training workflow** for a dataset you've
collected and labeled yourself outside the app (e.g. folders of
`Good_Smash`/`Poor_Smash`/`Good_Jump`/... clips) -- this is generally a
**better** training source, since the labels are real human judgement
from the start, not a rule-based suggestion corrected afterward.

### Folder layout

Place your videos under `dataset/raw_videos/`, one subfolder per class,
named `<Good|Poor>_<Footwork|Smash|Jump|Lunge>`:

```
dataset/raw_videos/
    Good_Footwork/*.mp4
    Poor_Footwork/*.mp4
    Good_Jump/*.mp4
    Poor_Jump/*.mp4
    Good_Lunge/*.mp4
    Poor_Lunge/*.mp4
    Good_Smash/*.mp4
    Poor_Smash/*.mp4
```

Folder names are matched case-insensitively, but consistent casing is
recommended. `.mp4`, `.avi`, `.mov`, `.mkv`, and `.ts` are all accepted
by this offline pipeline (it reads files directly with OpenCV) --
though note `.mkv` files using the **AV1** video codec are NOT
decodable by the pip-installed `opencv-python` build this project uses;
re-encode those to H.264 first (e.g.
`ffmpeg -i in.mkv -c:v libx264 out.mp4`) if you hit that.

### Two-command workflow

```bash
# 1. Verify the dataset + extract real ML features (needs a working
#    mediapipe install -- see requirements.txt's pinned version note)
python training/extract_dataset_features.py

# 2. Train + evaluate + save the Good/Poor technique-quality model
python training/train_dataset_model.py
```

**`training/extract_dataset_features.py`** re-uses `utils/video_utils.py`,
`modules/pose_estimation.py`, `modules/angle_calculator.py`, the four
movement-analysis modules, and `modules/feature_builder.py` --
unchanged, no duplicated logic. It writes:
- `reports/dataset_verification_report.json` -- class balance, naming
  issues, unsupported/duplicate files, per-file skip reasons.
- `dataset/extracted_features/labeled_dataset_features.csv` -- one row
  per successfully processed video, in the exact same flat feature
  shape used everywhere else in the app.

**`training/train_dataset_model.py`** trains a `RandomForestClassifier`
on that CSV (same algorithm/approach as `training/train_model.py`,
just a different data source and label set: `Good`/`Poor` instead of
the four risk levels). It writes:
- `models/model.joblib`, `models/scaler.joblib`, `models/model_metadata.json`
- `reports/model_evaluation.json` (accuracy/precision/recall/F1/confusion matrix)
- `reports/classification_report.txt`
- `dataset/train/train_split.csv`, `dataset/test/test_split.csv`

### Integration -- zero code changes needed

`modules/ml_predictor.py` doesn't hardcode any label vocabulary -- it
just loads whatever `model_metadata.json` describes and returns
whatever string the model predicts. Every template that shows
`result.ml_prediction.label` renders that string generically too. This
means switching from the placeholder risk-level labels to `Good`/`Poor`
requires **no changes anywhere else in the app** -- the dashboard's "AI
Model Status" section and every result page's "ML Model Prediction"
banner pick up the new model automatically the next time the app
process restarts (the model is cached in memory after first load; see
`modules/ml_predictor.py`'s `reset_cache()` if you retrain while the
app is still running).

### Important: run both commands on your own machine

This project's development/verification sandbox cannot run genuine
MediaPipe pose inference (see `requirements.txt`'s note on the pinned
`mediapipe==0.10.21` -- that exact version isn't fetchable in this
sandbox's restricted environment). Any model or CSV produced in that
sandbox is built on synthetic placeholder pose data and is **not**
meaningful for real predictions. Both commands above must be run in
your own properly-configured environment (`pip install -r
requirements.txt`, exactly as documented) to produce a real, trustworthy
model. No placeholder model or CSV is shipped in this project -- `models/`
and `dataset/extracted_features/` are intentionally empty until you run
them yourself.
