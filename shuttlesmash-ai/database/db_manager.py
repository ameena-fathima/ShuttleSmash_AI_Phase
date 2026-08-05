"""
database/db_manager.py
-----------------------
Handles everything related to *connecting* to the SQLite database.

Flask apps commonly use the pattern below:
  - `get_db()`   -> returns a connection, reusing one per request via
                    Flask's `g` object (so we don't open a new connection
                    for every query within the same request).
  - `close_db()` -> automatically closes that connection when the request
                    ends (registered as a "teardown" function).
  - `init_db()`  -> (re)builds all tables from schema.sql. Exposed as the
                    `flask init-db` command so the database can be created
                    with one command during setup.

No analysis logic lives here -- this file's only job is database plumbing.
"""

import sqlite3

import click
from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    """
    Return a SQLite connection for the current request.

    Flask's `g` object is a per-request storage area. The first time this
    function is called during a request we open a connection and stash it
    on `g`; subsequent calls within the same request reuse it.
    """
    if "db" not in g:
        g.db = sqlite3.connect(
            current_app.config["DATABASE_PATH"],
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        # Rows behave like dictionaries (row["username"]) instead of plain
        # tuples (row[0]) -- much easier to read in the rest of the code.
        g.db.row_factory = sqlite3.Row
        # Enforce foreign key constraints (SQLite disables this by default).
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(e=None):
    """Close the database connection at the end of the request, if one was opened."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Drop and recreate all tables by executing database/schema.sql."""
    db = get_db()
    with current_app.open_resource("database/schema.sql") as f:
        db.executescript(f.read().decode("utf8"))


@click.command("init-db")
@with_appcontext
def init_db_command():
    """
    CLI command: `flask init-db`
    Wipes and re-creates all tables. Run this once during setup, and again
    any time you want a completely fresh database.
    """
    init_db()
    click.echo("Database initialized (all tables created from schema.sql).")


def init_app(app):
    """
    Register database-related hooks on the Flask app.
    Called once from the application factory in app.py.
    """
    app.teardown_appcontext(close_db)   # auto-close DB connection after each request
    app.cli.add_command(init_db_command)  # enables `flask init-db`
