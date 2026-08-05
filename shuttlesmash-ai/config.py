"""
config.py
---------
Central place for all application configuration.

Keeping configuration in one file (instead of scattering constants across the
codebase) makes the app easy to reconfigure for different environments
(development vs. production) and easy to explain in a viva: "all settings
live in config.py".
"""

import os

# Absolute path to the project's root folder (the folder this file lives in).
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration used by the Flask app."""

    # --- Security ---
    # In production this MUST be overridden with an environment variable.
    # It is used by Flask to sign session cookies and flash messages.
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # --- Database ---
    # SQLite stores the whole database as a single file on disk.
    DATABASE_PATH = os.path.join(BASE_DIR, "instance", "shuttlesmash.db")

    # --- File storage locations (used from Phase 2 onward) ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    PROCESSED_FOLDER = os.path.join(BASE_DIR, "static", "processed")
    CHARTS_FOLDER = os.path.join(BASE_DIR, "static", "charts")
    REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")

    # --- Upload constraints (Phase 2) ---
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB maximum upload size
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}

    # --- Pose-estimation pipeline (Phase 2) ---
    # Keep 1 out of every N frames when sampling a video for pose detection.
    # Adjacent frames in a 30fps clip are nearly identical, so this speeds
    # up processing a lot without losing meaningful information.
    FRAME_SAMPLE_RATE = 5

    # --- Session behaviour ---
    # Session cookie expires when the browser is closed (safer for a shared/lab PC).
    SESSION_PERMANENT = False
