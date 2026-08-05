"""
services/video_service.py
---------------------------
Handles everything related to STORING an uploaded video: saving the file
to disk under a per-user folder, and creating/reading/updating its row in
the `videos` table.

This is the only layer allowed to touch the `videos` table directly --
routes in app.py call these functions instead of running SQL or file I/O
themselves, following the same pattern as services/auth_service.py from
Phase 1.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

from flask import current_app
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from database.db_manager import get_db


class VideoUploadError(Exception):
    """Raised for expected upload problems (bad filename, storage failure)."""
    pass


def save_video_file(file: FileStorage, user_id: int) -> Tuple[str, str]:
    """
    Save an uploaded video file to static/uploads/<user_id>/ on disk.

    The filename is passed through Werkzeug's secure_filename() to strip
    out anything unsafe (path separators, special/unicode characters), and
    a millisecond-timestamp prefix is added so uploading two files with
    the same original name never silently overwrites the first one.

    Args:
        file: the uploaded file from request.files.
        user_id: id of the currently logged-in user (used to namespace
            their upload folder from every other user's).

    Returns:
        (original_name, stored_path):
            original_name -- the sanitized filename, as the user knows it.
            stored_path -- absolute path where the file now lives on disk.

    Raises:
        VideoUploadError: if the filename is empty/unsafe after sanitizing.
    """
    original_name = secure_filename(file.filename)
    if not original_name:
        raise VideoUploadError("Invalid file name.")

    unique_name = f"{int(time.time() * 1000)}_{original_name}"

    user_folder = os.path.join(current_app.config["UPLOAD_FOLDER"], str(user_id))
    os.makedirs(user_folder, exist_ok=True)

    stored_path = os.path.join(user_folder, unique_name)
    file.save(stored_path)

    return original_name, stored_path


def create_video_record(
    user_id: int, original_name: str, stored_path: str, video_type: str
) -> int:
    """
    Insert a new row into the `videos` table with status 'uploaded'.

    Returns:
        The new video's auto-generated id.
    """
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO videos (user_id, original_name, stored_path, video_type, status)
        VALUES (?, ?, ?, ?, 'uploaded')
        """,
        (user_id, original_name, stored_path, video_type),
    )
    db.commit()
    return cursor.lastrowid


def update_video_status(video_id: int, status: str) -> None:
    """
    Update a video's processing status.
    Expected values: 'uploaded', 'processing', 'completed', 'failed'
    (matches the CHECK constraint on videos.status in schema.sql).
    """
    db = get_db()
    db.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))
    db.commit()


def get_video_by_id(video_id: int, user_id: int) -> Optional[Dict]:
    """
    Fetch a single video row, scoped to the given user_id.

    Scoping by user_id on every read is a simple but important security
    measure: it means one logged-in user can never view or trigger
    processing on another user's video, even if they guess/type another
    video's id in the URL.
    """
    db = get_db()
    row = db.execute(
        "SELECT * FROM videos WHERE id = ? AND user_id = ?", (video_id, user_id)
    ).fetchone()
    return dict(row) if row else None


def get_videos_for_user(user_id: int) -> List[Dict]:
    """Return all videos uploaded by a user, most recently uploaded first."""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM videos WHERE user_id = ? ORDER BY uploaded_at DESC",
        (user_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_video_for_user(user_id: int) -> Optional[Dict]:
    """Return the single most recently uploaded video for a user, or None
    if they haven't uploaded anything yet."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM videos WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def count_videos_for_user(user_id: int) -> int:
    """Return the total number of videos a user has uploaded."""
    db = get_db()
    row = db.execute(
        "SELECT COUNT(*) AS total FROM videos WHERE user_id = ?", (user_id,)
    ).fetchone()
    return row["total"]
