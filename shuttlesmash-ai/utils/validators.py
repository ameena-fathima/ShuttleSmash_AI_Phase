"""
utils/validators.py
--------------------
Small, framework-independent validation helpers.

These functions take plain strings/files and return either True/False or a
list of human-readable error messages. Keeping validation logic here
(instead of inline inside routes) means it can be unit-tested on its own
and reused across multiple forms.

Section 1: account registration/login validators (Phase 1)
Section 2: video upload validators (Phase 2)
"""

import os
import re
from typing import List, Set

from werkzeug.datastructures import FileStorage

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,20}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_username(username: str) -> bool:
    """Username: 3-20 characters, letters/numbers/underscore only."""
    return bool(USERNAME_PATTERN.match(username or ""))


def is_valid_email(email: str) -> bool:
    """Very simple email format check (not exhaustive, but catches typos)."""
    return bool(EMAIL_PATTERN.match(email or ""))


def is_valid_password(password: str) -> bool:
    """Password must be at least 6 characters long."""
    return bool(password) and len(password) >= 6


def validate_registration_form(full_name, username, email, password, confirm_password):
    """
    Validate all registration fields at once.

    Returns a list of error message strings. An empty list means the form
    is valid and safe to submit to the database.
    """
    errors = []

    if not full_name or not full_name.strip():
        errors.append("Full name is required.")

    if not is_valid_username(username):
        errors.append(
            "Username must be 3-20 characters and contain only letters, "
            "numbers, or underscores."
        )

    if not is_valid_email(email):
        errors.append("Please enter a valid email address.")

    if not is_valid_password(password):
        errors.append("Password must be at least 6 characters long.")

    if password != confirm_password:
        errors.append("Password and confirm password do not match.")

    return errors


def validate_login_form(username, password):
    """Validate login fields. Returns a list of error message strings."""
    errors = []
    if not username or not username.strip():
        errors.append("Username or email is required.")
    if not password:
        errors.append("Password is required.")
    return errors


# ---------------------------------------------------------------------
# Section 2: Video upload validators (Phase 2)
# ---------------------------------------------------------------------

def is_allowed_video_extension(filename: str, allowed_extensions: Set[str]) -> bool:
    """
    Check whether a filename ends in one of the allowed video extensions.
    Comparison is case-insensitive (".MP4" is treated the same as ".mp4").
    """
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in allowed_extensions


def get_file_size_bytes(file: FileStorage) -> int:
    """
    Determine an uploaded file's size in bytes.

    Werkzeug's FileStorage wraps a file-like stream that doesn't expose
    its size directly. Seeking to the end and reading the position (then
    seeking back to the start) is the standard way to measure it without
    reading the whole file into memory -- and rewinding afterwards is
    essential, otherwise the later `file.save(...)` call would write an
    empty file.
    """
    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    return size


def validate_video_upload(
    file: FileStorage, allowed_extensions: Set[str], max_size_bytes: int
) -> List[str]:
    """
    Validate an uploaded video file before it is saved to disk.

    Checks, in order:
        1. a file was actually selected (not an empty upload field)
        2. the file extension is one of `allowed_extensions`
        3. the file is not empty (0 bytes)
        4. the file does not exceed `max_size_bytes`

    Returns a list of human-readable error message strings. An empty list
    means the upload is valid and safe to save.
    """
    errors: List[str] = []

    if file is None or file.filename == "":
        errors.append("Please choose a video file to upload.")
        return errors  # nothing else to check if no file was chosen

    if not is_allowed_video_extension(file.filename, allowed_extensions):
        allowed = ", ".join(sorted(ext.upper() for ext in allowed_extensions))
        errors.append(f"Unsupported file type. Allowed formats: {allowed}.")

    size_bytes = get_file_size_bytes(file)
    if size_bytes == 0:
        errors.append("The selected file is empty.")
    elif size_bytes > max_size_bytes:
        max_mb = max_size_bytes // (1024 * 1024)
        errors.append(f"File is too large. Maximum allowed size is {max_mb} MB.")

    return errors
