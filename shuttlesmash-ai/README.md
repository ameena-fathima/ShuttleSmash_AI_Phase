# ShuttleSmash AI

AI-Based Badminton Performance Analysis and Injury Risk Detection System.
Final Year CS Project.

> **Status:** Phase 1 (auth), Phase 2 (video upload + pose estimation),
> Phase 3 (joint angle calculation + footwork/smash/jump/lunge movement
> analysis), Phase 4 (rule-based injury-risk detection + AI
> recommendations), and Phase 5 (ML-based injury-risk classifier, with
> automatic fallback to the Phase 4 rule-based engine when no model is
> trained) complete and verified. PDF report generation is added in a
> later phase. See `docs/ML_GUIDE.md` for the full ML dataset/training
> walkthrough.

## Tech Stack

Python 3.11 · Flask · SQLite · Bootstrap 5 · OpenCV · MediaPipe · NumPy ·
Pandas · Matplotlib · ReportLab

## 1. Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize the database (creates instance/shuttlesmash.db)
flask --app app init-db
```

> **Important — mediapipe version:** `requirements.txt` pins
> `mediapipe==0.10.21` on purpose. Newer mediapipe releases (0.10.22+)
> removed the legacy `mediapipe.solutions` API this project uses on some
> platforms, which causes `AttributeError: module 'mediapipe' has no
> attribute 'solutions'`. If you ever `pip install --upgrade mediapipe`
> and hit that error, just run `pip install mediapipe==0.10.21` again.

## 2. Run the app

```bash
flask --app app run --debug
```

Then open **http://127.0.0.1:5000** in your browser.

Alternatively:

```bash
python app.py
```

## 3. Project Structure (Phase 1 + Phase 2 + Phase 3 + Phase 4)

```
shuttlesmash-ai/
├── app.py                    # Flask app factory + all routes
├── config.py                  # Central configuration
├── database/
│   ├── db_manager.py          # SQLite connection handling + `flask init-db`
│   └── schema.sql             # All table definitions
├── modules/
│   ├── pose_estimation.py     # PoseEstimator: wraps MediaPipe Pose
│   ├── angle_calculator.py    # Joint angle math (elbow/shoulder/hip/knee/ankle)
│   ├── footwork_analysis.py   # Footwork metrics (lateral range, steps, stance, speed)
│   ├── smash_analysis.py      # Smash metrics (elbow extension, shoulder rotation, contact)
│   ├── jump_analysis.py       # Jump metrics (height %, landing knee angle, flight time)
│   ├── lunge_analysis.py      # Lunge metrics (knee flexion, knee-over-toe, torso lean)
│   ├── injury_risk.py         # Rule-based injury-risk detection (Phase 4)
│   ├── recommendation_engine.py # Turns risks into plain-English coaching tips (Phase 4)
│   └── feature_builder.py     # Flattens everything into one ML-ready feature row
├── services/
│   ├── auth_service.py        # Registration / login business logic
│   ├── video_service.py       # Video file storage + `videos` table access
│   └── analysis_service.py    # Full pipeline: pose -> angles -> movement -> save
├── utils/
│   ├── validators.py          # Form + video-upload validation
│   ├── decorators.py          # @login_required
│   └── video_utils.py         # OpenCV helpers (metadata, frame sampling)
├── templates/                 # Jinja2 HTML templates (Bootstrap 5)
├── static/
│   ├── uploads/<user_id>/     # Raw uploaded videos (gitignored)
│   └── processed/<user_id>/   # Skeleton-overlay preview images (gitignored)
└── instance/                  # SQLite database file lives here (gitignored)
```

See `docs/architecture.md` for the full planned structure across all
project phases.

## 4. Resetting the database

If you need a completely fresh database at any point during development:

```bash
flask --app app init-db
```

This drops and recreates every table, so all existing data (including
uploaded video records) is lost. Note: this does NOT delete the actual
video files under `static/uploads/` — remove those manually if needed.

## 5. Manual test flow

1. Go to `/register` and create an account.
2. Log in at `/login`.
3. Go to `/upload`, pick a short `.mp4`/`.avi`/`.mov` clip (under 100 MB)
   with a person clearly visible, choose a video type, and submit.
4. You'll land on the result page showing duration/FPS/resolution/frame
   counts and a skeleton-overlay preview image.
5. Visit `/dashboard` — it now shows your upload count and latest video
   with a link back to its result page.

