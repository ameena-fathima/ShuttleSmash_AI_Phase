"""
services/analysis_service.py
------------------------------
Orchestrates the full video-processing pipeline for a single uploaded video:

    Video (on disk)
        |
        v
    Read metadata + sample every Nth frame        (utils/video_utils.py)
        |
        v
    Pose detection on each sampled frame            (modules/pose_estimation.py)
        |
        v
    Joint angle calculation per frame                (modules/angle_calculator.py)
        |
        v
    Movement analysis (footwork/smash/jump/lunge)    (modules/*_analysis.py,
                                                        dispatched by video_type)
        |
        v
    Injury risk detection                             (modules/injury_risk.py)
        |
        v
    Recommendation generation                          (modules/recommendation_engine.py)
        |
        v
    Build the flat ML-ready feature row              (modules/feature_builder.py)
        |
        v
    Save results to the `analyses`, `injury_risks`,   (this file)
    and `recommendations` tables + a skeleton-overlay
    preview image
        |
        v
    Update the video's status in the DB               (services/video_service.py)
        |
        v
    Return a result summary for the results page

Phase 4 scope stops here -- risk detection and recommendations are both
explainable, rule-based logic (not machine learning). A trained model is
a later phase, and will be able to read training data straight out of
the `features_json` column this file writes (see modules/feature_builder.py).
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import cv2
from flask import current_app

from database.db_manager import get_db
from modules import angle_calculator, feature_builder, injury_risk, ml_predictor, recommendation_engine
from modules import footwork_analysis, jump_analysis, lunge_analysis, smash_analysis
from modules.pose_estimation import PoseEstimator
from services import video_service
from utils import video_utils

# Which movement-analysis module handles which video_type. "general"
# videos intentionally have no dedicated module -- joint angles are
# still calculated for them, but no footwork/smash/jump/lunge metrics.
_MOVEMENT_MODULES = {
    "footwork": footwork_analysis,
    "smash": smash_analysis,
    "jump": jump_analysis,
    "lunge": lunge_analysis,
}


class AnalysisError(Exception):
    """Raised when the processing pipeline fails for an expected reason
    (video not found, corrupted/unreadable file, etc.)."""
    pass


def _build_preview_path(user_id: int, video_id: int) -> str:
    """
    Build the absolute path where a video's skeleton-overlay preview
    image is stored: static/processed/<user_id>/<video_id>.jpg
    Using the video_id as the filename means re-processing a video always
    overwrites its old preview instead of leaving orphaned files behind.
    """
    folder = os.path.join(current_app.config["PROCESSED_FOLDER"], str(user_id))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{video_id}.jpg")


# ---------------------------------------------------------------------
# `analyses` table access (this service owns this table, the same way
# services/video_service.py owns the `videos` table).
# ---------------------------------------------------------------------

def save_analysis(
    video_id: int,
    user_id: int,
    angle_summary: Dict,
    movement_metrics: Dict,
    features: Dict,
    processed_preview: Optional[str],
    overall_score: Optional[float] = None,
) -> int:
    """
    Insert a new row into the `analyses` table.

    The rich, structured data (per-joint avg/min/max, movement metrics,
    and the full ML feature row) is stored as JSON text -- SQLite has no
    native nested/array column type, and JSON keeps this flexible as new
    metrics are added in later phases without needing new columns every
    time. See database/schema.sql for the column layout.

    `overall_score` (added in Phase 4) is a simple 0-100 rule-based
    safety score derived from detected injury risks -- see
    modules/injury_risk.py's calculate_overall_score().
    """
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO analyses (
            video_id, user_id, avg_joint_angles, movement_metrics_json,
            features_json, processed_preview, overall_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            user_id,
            json.dumps(angle_summary),
            json.dumps(movement_metrics),
            json.dumps(features),
            processed_preview,
            overall_score,
        ),
    )
    db.commit()
    return cursor.lastrowid


def save_injury_risks(analysis_id: int, risks: List[Dict]) -> None:
    """
    Insert every detected risk into the `injury_risks` table, linked to
    the analysis that produced them. Does nothing if the list is empty
    (no risks detected is a valid, common outcome).
    """
    if not risks:
        return
    db = get_db()
    db.executemany(
        """
        INSERT INTO injury_risks (
            analysis_id, risk_type, affected_joint, risk_level, confidence, description
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                analysis_id,
                risk["risk_type"],
                risk["affected_joint"],
                risk["risk_level"],
                risk["confidence"],
                risk["description"],
            )
            for risk in risks
        ],
    )
    db.commit()


def save_recommendations(analysis_id: int, recommendations: List[Dict]) -> None:
    """
    Insert every generated recommendation into the `recommendations`
    table, linked to the analysis that produced them. Does nothing if
    the list is empty (no risks -> no recommendations needed).
    """
    if not recommendations:
        return
    db = get_db()
    db.executemany(
        """
        INSERT INTO recommendations (analysis_id, category, recommendation, priority)
        VALUES (?, ?, ?, ?)
        """,
        [
            (analysis_id, rec["category"], rec["recommendation"], rec["priority"])
            for rec in recommendations
        ],
    )
    db.commit()


def get_injury_risks_for_analysis(analysis_id: int) -> List[Dict]:
    """Return every saved risk for an analysis, most severe first."""
    db = get_db()
    rows = db.execute(
        """
        SELECT * FROM injury_risks WHERE analysis_id = ?
        ORDER BY CASE risk_level WHEN 'High' THEN 0 WHEN 'Moderate' THEN 1 ELSE 2 END
        """,
        (analysis_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_recommendations_for_analysis(analysis_id: int) -> List[Dict]:
    """Return every saved recommendation for an analysis, highest priority first."""
    db = get_db()
    rows = db.execute(
        """
        SELECT * FROM recommendations WHERE analysis_id = ?
        ORDER BY CASE priority WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END
        """,
        (analysis_id,),
    ).fetchall()
    return [dict(row) for row in rows]


_RISK_LEVEL_RANK = {"High": 3, "Moderate": 2, "Low": 1}


def _highest_risk_level(risks: List[Dict]) -> Optional[str]:
    """Return the single most severe risk_level present in a list of
    risks (or dicts loaded from the injury_risks table), or None if the
    list is empty."""
    if not risks:
        return None
    return max(risks, key=lambda r: _RISK_LEVEL_RANK.get(r["risk_level"], 0))["risk_level"]


def get_latest_analysis_for_video(video_id: int) -> Optional[Dict]:
    """Return the most recent saved analysis row for a video, or None if
    it hasn't been analyzed yet."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM analyses WHERE video_id = ? ORDER BY analysis_date DESC LIMIT 1",
        (video_id,),
    ).fetchone()
    return dict(row) if row else None


def _result_from_saved_analysis(video: Dict, analysis: Dict) -> Dict[str, object]:
    """
    Rebuild a template-ready result dict from a previously saved analysis
    row, instead of re-running the whole pose-estimation pipeline every
    time someone revisits a result page. Video metadata (fps/duration/
    resolution/frame_count) is cheap to re-read from the file header, so
    that's recomputed fresh; the expensive part (pose detection + angle
    + movement analysis + risk detection) is loaded straight from the
    database.
    """
    info = video_utils.get_video_information(video["stored_path"])
    angle_summary = json.loads(analysis["avg_joint_angles"]) if analysis["avg_joint_angles"] else {}
    movement_metrics = (
        json.loads(analysis["movement_metrics_json"]) if analysis["movement_metrics_json"] else {}
    )
    features = json.loads(analysis["features_json"]) if analysis["features_json"] else {}
    risks = get_injury_risks_for_analysis(analysis["id"])
    recommendations = get_recommendations_for_analysis(analysis["id"])

    # Phase 5: try an ML prediction from the same stored feature row.
    # Returns None if no model is trained -- ml_prediction/prediction_source
    # simply reflect that; the existing rule-based `risks` list above is
    # completely unaffected either way.
    ml_prediction = ml_predictor.predict_risk_level(features)

    return {
        "fps": info["fps"],
        "duration_seconds": info["duration_seconds"],
        "resolution": info["resolution"],
        "frame_count": info["frame_count"],
        "sampled_frame_count": features.get("sampled_frame_count"),
        "person_detected_frame_count": features.get("person_detected_frame_count"),
        "preview_image_url": analysis["processed_preview"],
        "status": "completed",
        "angle_summary": angle_summary,
        "movement_metrics": movement_metrics,
        "video_type": video["video_type"],
        "analysis_id": analysis["id"],
        "overall_score": analysis["overall_score"],
        "risks": risks,
        "recommendations": recommendations,
        "highest_risk_level": _highest_risk_level(risks),
        "ml_prediction": ml_prediction,
        "prediction_source": "Machine Learning" if ml_prediction else "Rule-Based",
    }


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def process_video(
    video_id: int, user_id: int, force_reprocess: bool = False
) -> Dict[str, object]:
    """
    Run the full processing pipeline on one uploaded video, or return
    the previously saved result if it's already been processed (see
    `force_reprocess` below).

    Args:
        video_id: id of the row in the `videos` table to process.
        user_id: id of the user who owns the video (ownership is checked).
        force_reprocess: if False (default) and this video already has a
            saved analysis, that saved result is returned instantly
            instead of re-running pose estimation + angle/movement
            analysis again -- this is what makes revisiting a result
            page fast instead of re-processing the whole video every
            time. Pass True to force a fresh run (e.g. a future
            "Re-analyze" button).

    Returns:
        A dict describing the outcome:
            {
                "fps": float, "duration_seconds": float, "resolution": str,
                "frame_count": int, "sampled_frame_count": int,
                "person_detected_frame_count": int,
                "preview_image_url": str | None,
                "status": "completed",
                "angle_summary": dict,        # per-joint avg/min/max
                "movement_metrics": dict,     # video-type-specific metrics
                "video_type": str,
                "analysis_id": int,
                "overall_score": float,          # 0-100, see modules/injury_risk.py
                "risks": list[dict],             # detected injury risks
                "recommendations": list[dict],   # generated coaching tips
                "highest_risk_level": str | None,
            }

    Raises:
        AnalysisError: if the video record doesn't exist / isn't owned by
            this user, the video file can't be read, or no frames could
            be extracted from it. In every failure case, the video's
            status in the database is set to 'failed' before the error
            is raised, so the UI can reflect what happened.
    """
    video = video_service.get_video_by_id(video_id, user_id)
    if video is None:
        raise AnalysisError("Video not found.")

    # Fast path: reuse a previously saved analysis instead of re-running
    # the whole pipeline (which involves pose estimation on every
    # sampled frame and is the most expensive part of this app).
    if not force_reprocess and video["status"] == "completed":
        existing = get_latest_analysis_for_video(video_id)
        if existing is not None:
            return _result_from_saved_analysis(video, existing)

    video_service.update_video_status(video_id, "processing")

    try:
        # Step 1: read basic metadata (fps, duration, resolution, frame count).
        info = video_utils.get_video_information(video["stored_path"])

        # Step 2: sample every Nth frame instead of decoding the whole video.
        sample_rate = current_app.config.get("FRAME_SAMPLE_RATE", 5)
        sampled = video_utils.sample_frames(video["stored_path"], sample_rate=sample_rate)

        if not sampled:
            raise AnalysisError("No frames could be read from this video.")

        # Step 3: pose detection + joint angle calculation, frame by frame.
        # PoseEstimator is created ONCE and reused for every frame (see
        # modules/pose_estimation.py for why that matters for performance).
        frame_records = []          # one entry per frame WITH a detected person
        person_detected_count = 0
        best_frame = None           # clearest sampled frame, for the preview image
        best_frame_results = None
        best_visible_landmark_count = -1

        with PoseEstimator() as estimator:
            for frame_index, frame in sampled:
                results = estimator.detect(frame)
                landmarks = estimator.get_landmarks(results)

                if landmarks is None:
                    # Graceful handling of frames with no person visible --
                    # simply skip them and keep going.
                    continue

                person_detected_count += 1

                # Joint angles for this frame (Phase 3's core addition).
                angles = angle_calculator.calculate_all_angles(landmarks)
                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": round(frame_index / info["fps"], 3) if info["fps"] else 0.0,
                        "landmarks": {lm["name"]: lm for lm in landmarks},
                        "angles": angles,
                    }
                )

                # Track whichever sampled frame has the most clearly
                # visible landmarks, to use as the skeleton preview image.
                visible_landmark_count = sum(
                    1 for lm in landmarks if lm["visibility"] > 0.5
                )
                if visible_landmark_count > best_visible_landmark_count:
                    best_visible_landmark_count = visible_landmark_count
                    best_frame = frame
                    best_frame_results = results

            # Step 4: save a skeleton-overlay preview image, if a person
            # was detected in at least one sampled frame.
            preview_image_url = None
            if best_frame is not None:
                annotated_frame = estimator.draw_skeleton(best_frame, best_frame_results)
                preview_path = _build_preview_path(user_id, video_id)
                cv2.imwrite(preview_path, annotated_frame)
                relative_path = os.path.relpath(
                    preview_path, current_app.config["PROCESSED_FOLDER"]
                ).replace(os.sep, "/")
                preview_image_url = f"processed/{relative_path}"

        # Step 5: aggregate per-frame joint angles into avg/min/max per joint.
        angle_summary = angle_calculator.aggregate_angles(
            [f["angles"] for f in frame_records]
        )

        # Step 6: run the movement-analysis module matching this video's
        # type (footwork/smash/jump/lunge). "general" videos get no
        # movement-specific metrics, only joint angles.
        movement_module = _MOVEMENT_MODULES.get(video["video_type"])
        movement_metrics = (
            movement_module.analyze(frame_records, info["fps"]) if movement_module else {}
        )

        # Step 7: build the flat, ML-ready feature row (this is what a
        # later phase's CSV export will read straight out of the DB).
        video_info_for_features = {
            **info,
            "sampled_frame_count": len(sampled),
            "person_detected_frame_count": person_detected_count,
        }
        features = feature_builder.build_feature_row(
            video_id,
            user_id,
            video["video_type"],
            video_info_for_features,
            angle_summary,
            movement_metrics,
        )

        # Step 8: run rule-based injury-risk detection on the angles and
        # movement metrics computed above, then turn any detected risks
        # into plain-English recommendations. Both steps are explainable
        # rule-based logic (see modules/injury_risk.py and
        # modules/recommendation_engine.py) -- completely UNCHANGED from
        # Phase 4, so this always runs regardless of whether an ML model
        # exists.
        risks = injury_risk.detect_risks(angle_summary, movement_metrics, video["video_type"])
        recommendations = recommendation_engine.generate_recommendations(risks)
        overall_score = injury_risk.calculate_overall_score(risks)

        # Step 8b (Phase 5): additionally try an ML prediction from the
        # same feature row. This NEVER replaces the rule-based risks/
        # recommendations above -- it's shown alongside them as a
        # separate "AI Model Prediction". If no model has been trained
        # yet, predict_risk_level() returns None and the app behaves
        # exactly as it did in Phase 4.
        ml_prediction = ml_predictor.predict_risk_level(features)

        # Step 9: persist everything and mark the video as completed.
        analysis_id = save_analysis(
            video_id,
            user_id,
            angle_summary,
            movement_metrics,
            features,
            preview_image_url,
            overall_score,
        )
        save_injury_risks(analysis_id, risks)
        save_recommendations(analysis_id, recommendations)
        video_service.update_video_status(video_id, "completed")

        return {
            "fps": info["fps"],
            "duration_seconds": info["duration_seconds"],
            "resolution": info["resolution"],
            "frame_count": info["frame_count"],
            "sampled_frame_count": len(sampled),
            "person_detected_frame_count": person_detected_count,
            "preview_image_url": preview_image_url,
            "status": "completed",
            "angle_summary": angle_summary,
            "movement_metrics": movement_metrics,
            "video_type": video["video_type"],
            "analysis_id": analysis_id,
            "overall_score": overall_score,
            "ml_prediction": ml_prediction,
            "prediction_source": "Machine Learning" if ml_prediction else "Rule-Based",
            "risks": risks,
            "recommendations": recommendations,
            "highest_risk_level": _highest_risk_level(risks),
        }

    except Exception as exc:
        # Any failure anywhere in the pipeline above is caught here so we
        # can record 'failed' status instead of leaving the video stuck
        # on 'processing' forever.
        video_service.update_video_status(video_id, "failed")
        raise AnalysisError(f"Video processing failed: {exc}") from exc


def get_dashboard_summary_for_user(user_id: int) -> Optional[Dict[str, object]]:
    """
    Build the small set of stats the dashboard shows about a user's most
    recently analyzed video: overall score, how many risks were found,
    the single most severe risk level, and how many recommendations
    were generated.

    Returns None if the user hasn't uploaded a video yet, or their
    latest video hasn't been analyzed yet (still processing/failed) --
    the dashboard route/template handles that case with an empty state,
    same pattern as Phase 2's "no videos yet" message.
    """
    latest_video = video_service.get_latest_video_for_user(user_id)
    if latest_video is None:
        return None

    analysis = get_latest_analysis_for_video(latest_video["id"])
    if analysis is None:
        return None

    risks = get_injury_risks_for_analysis(analysis["id"])
    recommendations = get_recommendations_for_analysis(analysis["id"])

    # Phase 5: whether the latest video's prediction came from a trained
    # ML model or (as always, when no model exists) the rule-based
    # engine -- shown on the dashboard's ML status section.
    features = json.loads(analysis["features_json"]) if analysis["features_json"] else {}
    ml_prediction = ml_predictor.predict_risk_level(features)

    return {
        "overall_score": analysis["overall_score"],
        "risk_count": len(risks),
        "highest_risk_level": _highest_risk_level(risks),
        "recommendation_count": len(recommendations),
        "prediction_source": "Machine Learning" if ml_prediction else "Rule-Based",
    }


def get_dashboard_analytics_for_user(user_id: int, recent_limit: int = 6) -> Dict[str, object]:
    """
    Aggregate stats across ALL of a user's analyzed videos, for the
    redesigned dashboard's stat cards, charts, and recent-sessions list.

    This is a READ-ONLY aggregation over the existing `analyses`,
    `videos`, and `injury_risks` tables -- no schema change, and it
    does not alter how any individual video is analyzed (that pipeline
    is untouched). Per-session ML predictions are computed by calling
    modules/ml_predictor.py exactly the way _result_from_saved_analysis()
    already does, so this introduces no new prediction logic either.

    Returns a dict with:
        total_analyses, average_score, highest_score, high_risk_count,
        improvement_percent, score_trend (list, oldest->newest),
        risk_distribution ({"Low":n,"Moderate":n,"High":n}),
        weekly_activity (list of {"week","count"}, last 8 weeks),
        recent_sessions (list of dicts for the dashboard's session cards)
    """
    db = get_db()
    rows = db.execute(
        """
        SELECT a.id AS analysis_id, a.video_id, a.overall_score, a.analysis_date,
               a.processed_preview, a.features_json,
               v.original_name, v.video_type, v.status
        FROM analyses a
        JOIN videos v ON v.id = a.video_id
        WHERE a.user_id = ?
        ORDER BY a.analysis_date DESC
        """,
        (user_id,),
    ).fetchall()
    rows = [dict(r) for r in rows]

    empty_result = {
        "total_analyses": 0,
        "average_score": None,
        "highest_score": None,
        "high_risk_count": 0,
        "improvement_percent": None,
        "score_trend": [],
        "risk_distribution": {"Low": 0, "Moderate": 0, "High": 0},
        "weekly_activity": [],
        "recent_sessions": [],
    }
    if not rows:
        return empty_result

    scores = [r["overall_score"] for r in rows if r["overall_score"] is not None]
    average_score = round(sum(scores) / len(scores), 1) if scores else None
    highest_score = max(scores) if scores else None

    def _date_str(value) -> Optional[str]:
        """
        analyses.analysis_date is declared TIMESTAMP in schema.sql, so
        sqlite3's PARSE_DECLTYPES (enabled in database/db_manager.py)
        converts it into a real datetime.datetime object -- not a
        string. Templates/JS expect a plain "YYYY-MM-DD HH:MM:SS"
        string (the same format str(datetime) already produces), so
        this normalizes either case into that consistent string form.
        """
        if value is None:
            return None
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)

    # Risk distribution + high-risk session count -- one query for all
    # of this user's risk rows, instead of one query per analysis.
    analysis_ids = [r["analysis_id"] for r in rows]
    risk_distribution = {"Low": 0, "Moderate": 0, "High": 0}
    high_risk_analysis_ids = set()
    placeholders = ",".join("?" * len(analysis_ids))
    risk_rows = db.execute(
        f"SELECT analysis_id, risk_level FROM injury_risks WHERE analysis_id IN ({placeholders})",
        analysis_ids,
    ).fetchall()
    for rr in risk_rows:
        risk_distribution[rr["risk_level"]] = risk_distribution.get(rr["risk_level"], 0) + 1
        if rr["risk_level"] == "High":
            high_risk_analysis_ids.add(rr["analysis_id"])
    high_risk_count = len(high_risk_analysis_ids)

    # Improvement %: compares the average score of the earlier half vs.
    # the more-recent half of the user's scored analyses, chronologically
    # -- a simple, explainable trend measure. None if too few data points
    # exist for a meaningful comparison.
    chronological_scores = [
        r["overall_score"] for r in reversed(rows) if r["overall_score"] is not None
    ]
    improvement_percent = None
    if len(chronological_scores) >= 4:
        mid = len(chronological_scores) // 2
        earlier, later = chronological_scores[:mid], chronological_scores[mid:]
        earlier_avg = sum(earlier) / len(earlier)
        later_avg = sum(later) / len(later)
        if earlier_avg > 0:
            improvement_percent = round(((later_avg - earlier_avg) / earlier_avg) * 100, 1)

    # Score trend, oldest -> newest, for the dashboard's line chart.
    score_trend = [
        {"date": _date_str(r["analysis_date"]), "score": r["overall_score"]}
        for r in reversed(rows)
        if r["overall_score"] is not None
    ]

    # Weekly activity: number of analyses per ISO-ish week, last 8 weeks.
    week_counts = Counter()
    for r in rows:
        raw_date = r["analysis_date"]
        try:
            if hasattr(raw_date, "strftime"):
                parsed = raw_date
            else:
                parsed = datetime.fromisoformat(str(raw_date).split(".")[0])
        except (ValueError, AttributeError, TypeError):
            continue
        week_counts[parsed.strftime("%Y-W%W")] += 1
    weekly_activity = [
        {"week": k, "count": v} for k, v in sorted(week_counts.items())
    ][-8:]

    # Recent sessions, each with its ML prediction (reusing
    # modules/ml_predictor.py exactly as-is -- no changes to that module).
    recent_sessions = []
    for r in rows[:recent_limit]:
        features = json.loads(r["features_json"]) if r["features_json"] else {}
        prediction = ml_predictor.predict_risk_level(features)
        recent_sessions.append(
            {
                "video_id": r["video_id"],
                "name": r["original_name"],
                "video_type": r["video_type"],
                "status": r["status"],
                "date": _date_str(r["analysis_date"]),
                "score": r["overall_score"],
                "preview": r["processed_preview"],
                "prediction": prediction,
            }
        )

    return {
        "total_analyses": len(rows),
        "average_score": average_score,
        "highest_score": highest_score,
        "high_risk_count": high_risk_count,
        "improvement_percent": improvement_percent,
        "score_trend": score_trend,
        "risk_distribution": risk_distribution,
        "weekly_activity": weekly_activity,
        "recent_sessions": recent_sessions,
    }
