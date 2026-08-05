"""
app.py
------
Flask application entry point.

This file wires everything together but deliberately contains NO business
logic of its own:
  - Database plumbing lives in database/db_manager.py
  - Auth logic lives in services/auth_service.py
  - Form validation lives in utils/validators.py

Each route function is intentionally short: read input -> call a service
function -> render a template / redirect. This makes the app easy to read
top-to-bottom and easy to extend in later phases.

Phase 1 added: landing page, register/login/logout, dashboard placeholder.
Phase 2 added: video upload + pose-estimation processing pipeline routes.

Run with:
    flask --app app run --debug
or simply:
    python app.py
"""

import os

from flask import Flask, Response, flash, redirect, render_template, request, session, url_for
from config import Config
from database import db_manager
from modules import ml_predictor, report_generator
from services import auth_service, video_service, analysis_service
from services.auth_service import AuthError
from services.analysis_service import AnalysisError, process_video
from services.video_service import VideoUploadError
from utils.decorators import login_required
from utils.validators import (
    validate_login_form,
    validate_registration_form,
    validate_video_upload,
)


def create_app():
    """
    Application factory.

    Using a factory function (instead of creating the Flask app directly
    at import time) is standard practice: it lets us create fresh app
    instances for automated tests, and keeps configuration explicit.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # Make sure the folders we depend on actually exist (instance/ holds
    # the SQLite file; static/uploads and static/processed are used by
    # the Phase 2 upload + pose-estimation pipeline below).
    for folder in [
        os.path.dirname(app.config["DATABASE_PATH"]),
        app.config["UPLOAD_FOLDER"],
        app.config["PROCESSED_FOLDER"],
        app.config["CHARTS_FOLDER"],
        app.config["REPORTS_FOLDER"],
    ]:
        os.makedirs(folder, exist_ok=True)

    # Register database teardown + the `flask init-db` CLI command.
    db_manager.init_app(app)

    # ------------------------------------------------------------------
    # Public pages
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        """Landing page. Sends logged-in users straight to their dashboard."""
        if session.get("user_id"):
            return redirect(url_for("dashboard"))
        return render_template("index.html")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @app.route("/register", methods=["GET", "POST"])
    def register():
        """Create a new account."""
        if session.get("user_id"):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            full_name = request.form.get("full_name", "")
            username = request.form.get("username", "")
            email = request.form.get("email", "")
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            # Step 1: field-level validation (format, matching passwords, etc.)
            errors = validate_registration_form(
                full_name, username, email, password, confirm_password
            )

            # Step 2: only hit the database if the form itself is valid.
            if not errors:
                try:
                    auth_service.register_user(full_name, username, email, password)
                except AuthError as e:
                    errors.append(str(e))

            if errors:
                for message in errors:
                    flash(message, "danger")
                # Re-render the form, keeping what the user already typed
                # (except passwords, which are never echoed back).
                return render_template(
                    "auth/register.html",
                    full_name=full_name,
                    username=username,
                    email=email,
                )

            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))

        return render_template("auth/register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        """Log an existing user in."""
        if session.get("user_id"):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            identifier = request.form.get("identifier", "")
            password = request.form.get("password", "")

            errors = validate_login_form(identifier, password)

            if not errors:
                try:
                    user = auth_service.authenticate_user(identifier, password)
                    # Only the user id is stored in the session; everything
                    # else about the user is looked up fresh when needed.
                    session.clear()
                    session["user_id"] = user["id"]
                    session["username"] = user["username"]
                    flash(f"Welcome back, {user['full_name']}!", "success")
                    return redirect(url_for("dashboard"))
                except AuthError as e:
                    errors.append(str(e))

            for message in errors:
                flash(message, "danger")
            return render_template("auth/login.html", identifier=identifier)

        return render_template("auth/login.html")

    @app.route("/logout")
    def logout():
        """Clear the session and log the user out."""
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("index"))

    # ------------------------------------------------------------------
    # Dashboard (placeholder for Phase 1 -- fleshed out in Phase 6)
    # ------------------------------------------------------------------

    @app.route("/dashboard")
    @login_required
    def dashboard():
        """
        Logged-in landing page.
        Shows how many videos the user has uploaded, their most recent
        upload's processing status, a summary of the latest analysis's
        injury-risk findings (Phase 4), and (Phase 5) the current ML
        model's status -- trained or not, version, training date,
        accuracy, and whether the latest prediction came from the model
        or the rule-based fallback.
        """
        user = auth_service.get_user_by_id(session["user_id"])
        total_videos = video_service.count_videos_for_user(session["user_id"])
        latest_video = video_service.get_latest_video_for_user(session["user_id"])
        analysis_summary = analysis_service.get_dashboard_summary_for_user(session["user_id"])
        model_status = ml_predictor.get_model_status()
        dashboard_analytics = analysis_service.get_dashboard_analytics_for_user(session["user_id"])
        return render_template(
            "dashboard.html",
            user=user,
            total_videos=total_videos,
            latest_video=latest_video,
            analysis_summary=analysis_summary,
            model_status=model_status,
            dashboard_analytics=dashboard_analytics,
        )

    # ------------------------------------------------------------------
    # Video upload & pose-estimation pipeline (Phase 2)
    # ------------------------------------------------------------------

    @app.route("/upload", methods=["GET", "POST"])
    @login_required
    def upload():
        """
        Upload a badminton video and run it through the Phase 2 pipeline:
        frame sampling + MediaPipe pose detection + skeleton preview.

        On success, renders the result page directly (rather than
        redirecting) so the user immediately sees their video's metadata,
        skeleton preview, and processing status without an extra click.
        """
        if request.method == "POST":
            video_file = request.files.get("video_file")
            video_type = request.form.get("video_type", "general")

            # Step 1: validate the file itself (extension, size, emptiness)
            # before touching the disk or the database at all.
            errors = validate_video_upload(
                video_file,
                app.config["ALLOWED_VIDEO_EXTENSIONS"],
                app.config["MAX_CONTENT_LENGTH"],
            )
            if errors:
                for message in errors:
                    flash(message, "danger")
                return render_template("upload.html")

            try:
                # Step 2: save the file to disk and record it in the DB.
                original_name, stored_path = video_service.save_video_file(
                    video_file, session["user_id"]
                )
                video_id = video_service.create_video_record(
                    session["user_id"], original_name, stored_path, video_type
                )

                # Step 3: run the pose-estimation pipeline on the saved file.
                result = process_video(video_id, session["user_id"])

                flash("Video uploaded and processed successfully.", "success")
                video = video_service.get_video_by_id(video_id, session["user_id"])
                return render_template(
                    "upload_result.html", video=video, result=result
                )

            except VideoUploadError as e:
                flash(str(e), "danger")
                return render_template("upload.html")
            except AnalysisError as e:
                # The file was saved, but processing failed (e.g. corrupted
                # video). Show the failure clearly instead of a blank error.
                flash(str(e), "warning")
                return redirect(url_for("dashboard"))

        return render_template("upload.html")

    @app.route("/videos/<int:video_id>")
    @login_required
    def view_video_result(video_id: int):
        """
        Re-display the processing result for a previously uploaded video
        (e.g. reached via the dashboard's "latest upload" link). Metadata
        is recomputed from the video file on disk rather than stored
        redundantly in the database, per the Phase 2 database rule of
        only using the existing `videos` table.
        """
        video = video_service.get_video_by_id(video_id, session["user_id"])
        if video is None:
            flash("Video not found.", "danger")
            return redirect(url_for("dashboard"))

        try:
            result = process_video(video_id, session["user_id"])
        except AnalysisError as e:
            flash(str(e), "warning")
            return redirect(url_for("dashboard"))

        video = video_service.get_video_by_id(video_id, session["user_id"])
        return render_template("upload_result.html", video=video, result=result)

    @app.route("/videos/<int:video_id>/report")
    @login_required
    def download_video_report(video_id: int):
        video = video_service.get_video_by_id(video_id, session["user_id"])
        if video is None:
            flash("Video not found.", "danger")
            return redirect(url_for("dashboard"))

        try:
            result = process_video(video_id, session["user_id"])
        except AnalysisError as e:
            flash(str(e), "warning")
            return redirect(url_for("dashboard"))

        pdf_bytes = report_generator.generate_pdf_report(video, result)
        safe_name = "".join(
            c for c in video["original_name"]
            if c.isalnum() or c in "._-"
        ) or "video"

        filename = f'ShuttleSmash_Report_{safe_name}.pdf'

        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={
                "Content-Disposition":
                f'attachment; filename="{filename}"'
            },
        )
    
    @app.route("/history")
    @login_required
    def history():
        """Show all videos uploaded by the logged-in user."""
        user = auth_service.get_user_by_id(session["user_id"])
        videos = video_service.get_videos_for_user(session["user_id"])

        return render_template(
            "history.html",
            user=user,
            videos=videos,
        )

    @app.errorhandler(413)
    def file_too_large(e):
        """
        Flask/Werkzeug automatically rejects any request larger than
        MAX_CONTENT_LENGTH with a raw 413 response. This handler turns
        that into a friendly flash message instead of a blank error page.
        """
        flash(
            f"File is too large. Maximum allowed size is "
            f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB.",
            "danger",
        )
        return redirect(url_for("upload"))

    return app


# Allows `python app.py` as an alternative to `flask run`.
if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(debug=True)
else:
    # Allows `flask --app app run` / `flask init-db` to find the app.
    app = create_app()
