"""
modules/report_generator.py
------------------------------
NEW FILE -- adds PDF report generation as a pure presentation/export
layer on top of the EXISTING, already-computed analysis result.

This module does NOT:
  - run pose estimation
  - run any movement/joint-angle analysis
  - query the database directly
  - call the ML model
  - compute injury risk or the safety score
  - touch modules/injury_risk.py, modules/*_analysis.py,
    modules/angle_calculator.py, modules/ml_predictor.py,
    modules/feature_builder.py, or services/analysis_service.py in any way

It only takes the `video` dict (a row from the `videos` table) and the
`result` dict -- the EXACT same dict services/analysis_service.py's
process_video() already returns and templates/upload_result.html
already renders -- and lays it out as a PDF using ReportLab. If a value
is missing/None in `result`, the PDF shows "N/A" rather than guessing;
nothing is recomputed or fabricated here.

Usage (see app.py's new /videos/<id>/report route):
    from modules.report_generator import generate_pdf_report
    pdf_bytes = generate_pdf_report(video, result)
"""

import io
from datetime import datetime, timezone
from typing import Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Brand colors, matched to the app's existing dark-theme accent palette
# (static/css/style.css's --court-green / --danger / --warning) so the
# PDF feels like the same product, not a generic report template.
_COURT_GREEN = colors.HexColor("#0f9d6e")
_DANGER = colors.HexColor("#d9364f")
_WARNING = colors.HexColor("#c98a12")
_INK = colors.HexColor("#1a1f27")
_MUTED = colors.HexColor("#5b6472")
_LIGHT_BG = colors.HexColor("#f4f6f8")

_RISK_COLOR = {"High": _DANGER, "Moderate": _WARNING, "Low": _COURT_GREEN}


def _styles():
    """Build the small set of paragraph styles this report reuses."""
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], textColor=_COURT_GREEN, fontSize=22, spaceAfter=2
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], textColor=_MUTED, fontSize=10, spaceAfter=14
        ),
        "h2": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            textColor=_INK,
            fontSize=13,
            spaceBefore=16,
            spaceAfter=6,
        ),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontSize=9.5, leading=13, textColor=_INK),
        "muted": ParagraphStyle("Muted", parent=base["Normal"], fontSize=8.5, textColor=_MUTED),
        "cell": ParagraphStyle("Cell", parent=base["Normal"], fontSize=9, leading=12, textColor=_INK),
    }
    return styles


def _fmt(value, suffix: str = "") -> str:
    """Render a possibly-missing value as a display string, never guessing."""
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def _table(data, col_widths=None, header=True):
    """Shared table styling used by every section below."""
    style_cmds = [
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dfe3e8")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, _LIGHT_BG]),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), _COURT_GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle(style_cmds))
    return t


def _derive_summary(result: Dict) -> Dict[str, object]:
    """
    Build the same honest, non-fabricated Strengths / Needs Improvement /
    Rating summary templates/upload_result.html already shows on-screen
    -- re-derived here in Python (from result["risks"]/["video_type"]/
    ["overall_score"], all already-computed values) so the PDF and the
    web page always agree, without importing Jinja into a PDF builder.
    Nothing new is inferred; this only reorganizes existing risk data.
    """
    risks = result.get("risks") or []
    risk_types = {r["risk_type"] for r in risks}
    video_type = result.get("video_type")
    score = result.get("overall_score")

    strengths = []
    if result.get("angle_summary"):
        if "Knee Valgus Risk" not in risk_types:
            strengths.append("Balanced knee alignment")
        if "Asymmetrical Loading" not in risk_types:
            strengths.append("Even left/right loading")
        if video_type == "jump" and "Poor Landing Technique" not in risk_types:
            strengths.append("Stable landing mechanics")
        if video_type == "smash" and "Shoulder Overextension" not in risk_types:
            strengths.append("Controlled shoulder extension")
        if video_type == "smash" and "Smash Technique Load Risk" not in risk_types:
            strengths.append("Efficient arm extension and trunk rotation")
        if video_type == "lunge" and "Lunge Overreach" not in risk_types:
            strengths.append("Good lunge depth control")
        if not risks:
            strengths.append("No injury risks detected")

    needs_improvement = [r["risk_type"] for r in risks] or []

    if score is None:
        rating = "Not enough data"
    elif score >= 85:
        rating = "Advanced Athlete"
    elif score >= 70:
        rating = "Intermediate Athlete"
    elif score >= 50:
        rating = "Developing Athlete"
    else:
        rating = "Needs Attention"

    return {"strengths": strengths, "needs_improvement": needs_improvement, "rating": rating}


def generate_pdf_report(video: Dict, result: Dict) -> bytes:
    """
    Build a PDF report from an already-computed analysis.

    Args:
        video: a row dict from the `videos` table (as returned by
            services/video_service.get_video_by_id()) -- original_name,
            video_type, uploaded_at, status.
        result: the result dict services/analysis_service.py's
            process_video() returns (same shape rendered by
            templates/upload_result.html) -- duration_seconds, fps,
            resolution, frame_count, sampled_frame_count,
            person_detected_frame_count, ml_prediction, overall_score,
            risks, recommendations, movement_metrics, angle_summary,
            video_type, highest_risk_level.

    Returns:
        The PDF file's raw bytes, ready to be sent as a Flask response.
    """
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        title=f"ShuttleSmash AI Report - {video.get('original_name', 'video')}",
    )
    story = []

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------
    story.append(Paragraph("ShuttleSmash AI", styles["title"]))
    story.append(Paragraph("Badminton Performance &amp; Injury-Risk Report", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=_COURT_GREEN, spaceAfter=10))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header_table = _table(
        [
            ["Video File", _fmt(video.get("original_name"))],
            ["Video Type", _fmt((video.get("video_type") or "").capitalize())],
            ["Uploaded", _fmt(video.get("uploaded_at"))],
            ["Report Generated", generated_at],
        ],
        col_widths=[1.6 * inch, 4.65 * inch],
        header=False,
    )
    story.append(header_table)

    # ---------------------------------------------------------------
    # Video metadata
    # ---------------------------------------------------------------
    story.append(Paragraph("Video Metadata", styles["h2"]))
    meta_rows = [
        ["Metric", "Value"],
        ["Duration", _fmt(result.get("duration_seconds"), " sec")],
        ["Frame Rate (FPS)", _fmt(result.get("fps"))],
        ["Resolution", _fmt(result.get("resolution"))],
        ["Total Frames", _fmt(result.get("frame_count"))],
        ["Sampled Frames", _fmt(result.get("sampled_frame_count"))],
        [
            "Frames with Person Detected",
            f"{_fmt(result.get('person_detected_frame_count'))} / {_fmt(result.get('sampled_frame_count'))}",
        ],
    ]
    story.append(_table(meta_rows, col_widths=[2.5 * inch, 3.75 * inch]))

    # ---------------------------------------------------------------
    # Technique classification (ML) + Safety score
    # ---------------------------------------------------------------
    story.append(Paragraph("Technique Classification &amp; Safety Score", styles["h2"]))
    ml_prediction = result.get("ml_prediction")
    if ml_prediction:
        classification_text = (
            f"<b>{ml_prediction.get('label', 'N/A')}</b> "
            f"({_fmt(ml_prediction.get('confidence'), '%')} confidence, "
            f"model v{_fmt(ml_prediction.get('model_version'))}, source: Machine Learning)"
        )
    else:
        classification_text = "No trained ML model available -- rule-based results only."

    score = result.get("overall_score")
    score_text = f"<b>{_fmt(score)}</b> / 100" if score is not None else "N/A"

    summary_rows = [
        ["Technique Classification", Paragraph(classification_text, styles["cell"])],
        ["Overall Safety Score", Paragraph(score_text, styles["cell"])],
        ["Highest Risk Level", _fmt(result.get("highest_risk_level"), "") if result.get("highest_risk_level") else "None"],
    ]
    story.append(_table(summary_rows, col_widths=[2.2 * inch, 4.05 * inch], header=False))

    # ---------------------------------------------------------------
    # Injury risk assessment
    # ---------------------------------------------------------------
    story.append(Paragraph("Injury Risk Assessment", styles["h2"]))
    risks = result.get("risks") or []
    if risks:
        risk_rows = [["Risk Type", "Level", "Joint", "Conf.", "Description"]]
        for r in risks:
            risk_rows.append(
                [
                    Paragraph(r.get("risk_type", ""), styles["cell"]),
                    r.get("risk_level", ""),
                    r.get("affected_joint", ""),
                    _fmt(r.get("confidence"), "%"),
                    Paragraph(r.get("description", ""), styles["cell"]),
                ]
            )
        risk_table = _table(
            risk_rows, col_widths=[1.15 * inch, 0.6 * inch, 0.85 * inch, 0.5 * inch, 3.15 * inch]
        )
        # Color the "Level" cell per row to match the app's risk badges.
        style_cmds = []
        for i, r in enumerate(risks, start=1):
            color = _RISK_COLOR.get(r.get("risk_level"))
            if color:
                style_cmds.append(("TEXTCOLOR", (1, i), (1, i), color))
                style_cmds.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
        if style_cmds:
            risk_table.setStyle(TableStyle(style_cmds))
        story.append(risk_table)
    else:
        story.append(
            Paragraph(
                "No injury risks were detected based on the current rule-based checks."
                if result.get("angle_summary")
                else "No injury risks could be checked -- no person was detected in this video.",
                styles["body"],
            )
        )

    # ---------------------------------------------------------------
    # Coaching recommendations
    # ---------------------------------------------------------------
    story.append(Paragraph("Coaching Recommendations", styles["h2"]))
    recommendations = result.get("recommendations") or []
    if recommendations:
        rec_rows = [["Category", "Priority", "Recommendation"]]
        for rec in recommendations:
            rec_rows.append(
                [
                    rec.get("category", ""),
                    rec.get("priority", ""),
                    Paragraph(rec.get("recommendation", ""), styles["cell"]),
                ]
            )
        story.append(_table(rec_rows, col_widths=[1.2 * inch, 0.9 * inch, 4.15 * inch]))
    else:
        story.append(Paragraph("No recommendations to show for this video.", styles["body"]))

    # ---------------------------------------------------------------
    # Movement metrics (video-type specific)
    # ---------------------------------------------------------------
    movement_metrics = result.get("movement_metrics") or {}
    if movement_metrics:
        story.append(
            Paragraph(f"{(result.get('video_type') or '').capitalize()} Movement Metrics", styles["h2"])
        )
        mm_rows = [["Metric", "Value"]]
        for key, value in movement_metrics.items():
            mm_rows.append([key.replace("_", " ").capitalize(), _fmt(value)])
        story.append(_table(mm_rows, col_widths=[3.0 * inch, 3.25 * inch]))

    # ---------------------------------------------------------------
    # Joint angles
    # ---------------------------------------------------------------
    angle_summary = result.get("angle_summary") or {}
    if angle_summary:
        story.append(Paragraph("Joint Angles (degrees)", styles["h2"]))
        angle_rows = [["Joint", "Avg", "Min", "Max"]]
        for name, stats in angle_summary.items():
            angle_rows.append(
                [
                    name.replace("_", " ").capitalize(),
                    _fmt(stats.get("avg")),
                    _fmt(stats.get("min")),
                    _fmt(stats.get("max")),
                ]
            )
        story.append(_table(angle_rows, col_widths=[2.6 * inch, 1.2 * inch, 1.2 * inch, 1.25 * inch]))

    # ---------------------------------------------------------------
    # AI analysis summary
    # ---------------------------------------------------------------
    story.append(Paragraph("AI Analysis Summary", styles["h2"]))
    summary = _derive_summary(result)
    if summary["strengths"]:
        story.append(Paragraph("<b>Strengths</b>", styles["body"]))
        for s in summary["strengths"]:
            story.append(Paragraph(f"&#10003; {s}", styles["body"]))
    if summary["needs_improvement"]:
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Needs Improvement</b>", styles["body"]))
        for n in summary["needs_improvement"]:
            story.append(Paragraph(f"&bull; {n}", styles["body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Overall Rating:</b> {summary['rating']}", styles["body"]))

    # ---------------------------------------------------------------
    # Footer disclaimer
    # ---------------------------------------------------------------
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#dfe3e8")))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Injury-risk checks are explainable, rule-based logic derived from joint angles and "
            "movement metrics, and are supplemented by an optional trained ML technique classifier "
            "when available. Thresholds used are project heuristics for a college demonstration "
            "project, not clinically validated medical guidance. Generated by ShuttleSmash AI.",
            styles["muted"],
        )
    )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
