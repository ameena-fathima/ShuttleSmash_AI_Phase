"""
services/auth_service.py
-------------------------
Business logic for registration and login.

This is the *only* layer allowed to talk to both the database (via
db_manager.get_db) and password-hashing utilities. Flask routes in app.py
call these functions instead of running SQL or hashing passwords directly
-- this keeps app.py short and keeps all "how do we authenticate a user"
logic in one place.
"""

from werkzeug.security import check_password_hash, generate_password_hash

from database.db_manager import get_db


class AuthError(Exception):
    """Raised for expected authentication problems (duplicate user, wrong
    password, etc.) so routes can show a friendly message instead of a
    generic server error."""
    pass


def username_or_email_exists(username: str, email: str) -> bool:
    """Check whether a user with this username or email already exists."""
    db = get_db()
    row = db.execute(
        "SELECT id FROM users WHERE username = ? OR email = ?",
        (username, email),
    ).fetchone()
    return row is not None


def register_user(full_name: str, username: str, email: str, password: str) -> int:
    """
    Create a new user account.

    The plain-text password is never stored -- only a salted hash produced
    by werkzeug's generate_password_hash. Returns the new user's id.
    Raises AuthError if the username/email is already taken.
    """
    if username_or_email_exists(username, email):
        raise AuthError("An account with that username or email already exists.")

    password_hash = generate_password_hash(password)

    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO users (full_name, username, email, password_hash)
        VALUES (?, ?, ?, ?)
        """,
        (full_name.strip(), username.strip(), email.strip().lower(), password_hash),
    )
    db.commit()
    return cursor.lastrowid


def authenticate_user(identifier: str, password: str) -> dict:
    """
    Verify a login attempt.

    `identifier` may be either a username or an email address, so users
    can log in with whichever they remember. Returns the user row (as a
    dict) on success. Raises AuthError on any failure -- the message is
    deliberately generic ("Invalid username/email or password") so we
    don't reveal whether the username or the password was wrong, which is
    a basic security best-practice worth mentioning in a viva.
    """
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (identifier.strip(), identifier.strip().lower()),
    ).fetchone()

    if user is None or not check_password_hash(user["password_hash"], password):
        raise AuthError("Invalid username/email or password.")

    return dict(user)


def get_user_by_id(user_id: int):
    """Fetch a single user by id (used to load the logged-in user's info)."""
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(user) if user else None
