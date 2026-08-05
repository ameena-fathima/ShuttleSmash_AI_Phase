"""
utils/decorators.py
--------------------
Reusable route decorators.

`login_required` protects a Flask view so that it can only be accessed by a
logged-in user. Any route decorated with it will redirect anonymous
visitors to the login page instead of showing the page content.
"""

from functools import wraps

from flask import flash, redirect, session, url_for


def login_required(view_func):
    """
    Decorator: place @login_required directly above any route function
    that should only be visible to logged-in users, e.g.:

        @app.route("/dashboard")
        @login_required
        def dashboard():
            ...
    """

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if session.get("user_id") is None:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view
