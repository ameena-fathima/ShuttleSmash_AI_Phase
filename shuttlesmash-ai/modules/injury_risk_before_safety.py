"""
modules/injury_risk.py
------------------------
Rule-based (NOT machine learning) injury-risk detection for badminton
players, built entirely on top of the joint angles and movement metrics
already computed in Phase 3 (modules/angle_calculator.py and the
footwork/smash/jump/lunge analysis modules).

Why rules instead of ML? Two reasons, both worth explaining in a viva:
  1. No labeled injury dataset exists yet to train a classifier on --
     that's exactly what Phase 3's modules/feature_builder.py is
     preparing the ground for. Once enough analyzed videos exist, a
     later phase can revisit this with a trained Random Forest.
  2. Rule-based thresholds are fully explainable: every flagged risk can
     be traced back to one specific number crossing one specific,
     documented threshold -- exactly what a viva panel will want
     justified.

IMPORTANT LIMITATION (stated here, not hidden): every angle in this app
is measured in 2D from a single, uncalibrated camera. True clinical
"knee valgus" is a frontal-plane (side-to-side) measurement; without a
second camera or calibration, this module uses LEFT/RIGHT ASYMMETRY as
an explainable proxy for valgus-type loading patterns instead of
claiming a real frontal-plane angle. This limitation is repeated in each
risk's own description so nothing here overclaims clinical accuracy.
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------
# Thresholds -- kept as named constants (not magic numbers buried in
# logic) so they're easy to find, justify in a viva, and tune later.
# ---------------------------------------------------------------------

KNEE_ASYMMETRY_LOW_DEG = 15.0
KNEE_ASYMMETRY_HIGH_DEG = 25.0

POOR_LANDING_KNEE_ANGLE_LOW_DEG = 160.0   # a near-straight knee absorbs less shock
POOR_LANDING_KNEE_ANGLE_HIGH_DEG = 172.0

SHOULDER_OVEREXTENSION_LOW_DEG = 170.0
SHOULDER_OVEREXTENSION_HIGH_DEG = 178.0

LUNGE_KNEE_OVER_TOE_LOW = 0.06   # normalized image-width units
LUNGE_KNEE_OVER_TOE_HIGH = 0.12

HIP_KNEE_ASYMMETRY_LOW_DEG = 12.0
HIP_KNEE_ASYMMETRY_HIGH_DEG = 20.0

# Smash-specific technique-risk heuristics.
# These are project heuristics, NOT clinical thresholds.
SMASH_ELBOW_EXTENSION_FLOOR_DEG = 150.0
SMASH_ELBOW_EXTENSION_DEFICIT_LOW_DEG = 10.0
SMASH_ELBOW_EXTENSION_DEFICIT_HIGH_DEG = 35.0

SMASH_SHOULDER_ROTATION_FLOOR_DEG = 25.0
SMASH_ROTATION_DEFICIT_LOW_DEG = 5.0
SMASH_ROTATION_DEFICIT_HIGH_DEG = 18.0

# Used by calculate_overall_score() -- how many points a detected risk
# subtracts from a starting score of 100, based on its severity.
_SCORE_PENALTY = {"High": 30, "Moderate": 15, "Low": 5}


def _get_avg(angle_summary: Dict, key: str) -> Optional[float]:
    """Safely pull an angle's average value out of the aggregate dict
    built by angle_calculator.aggregate_angles(), or None if missing."""
    stats = angle_summary.get(key)
    return stats.get("avg") if stats else None


def _level_and_confidence(value: float, low: float, high: float) -> Optional[Dict]:
    """
    Shared helper: turn one raw measurement into a (risk_level,
    confidence) pair given a "Low" threshold and a "High" threshold.

    - Below `low`: not flagged at all (returns None).
    - Between `low` and `high`: "Low" or "Moderate", depending on how
      far past `low` the value has gotten.
    - At or above `high`: "High".

    `confidence` is a simple, transparent 0-100 score reflecting how far
    past the threshold the value sits (capped at 100) -- NOT a
    statistical probability, just an explainable "how clearly did this
    cross the line" indicator appropriate for a rule-based system.
    """
    if value < low:
        return None

    span = max(high - low, 0.01)
    progress = min((value - low) / span, 1.5)  # allow some overshoot past "high"

    if value >= high:
        level = "High"
    elif progress >= 0.5:
        level = "Moderate"
    else:
        level = "Low"

    confidence = round(min(50 + progress * 50, 100), 1)
    return {"risk_level": level, "confidence": confidence}


def _check_knee_valgus(angle_summary: Dict) -> Optional[Dict]:
    """
    Proxy check: sustained left/right KNEE angle asymmetry across the
    video. See this module's docstring for why asymmetry (not a true
    frontal-plane angle) is used as the signal here.
    """
    left = _get_avg(angle_summary, "left_knee_angle")
    right = _get_avg(angle_summary, "right_knee_angle")
    if left is None or right is None:
        return None

    diff = abs(left - right)
    verdict = _level_and_confidence(diff, KNEE_ASYMMETRY_LOW_DEG, KNEE_ASYMMETRY_HIGH_DEG)
    if verdict is None:
        return None

    return {
        "risk_type": "Knee Valgus Risk",
        "affected_joint": "Knee",
        "risk_level": verdict["risk_level"],
        "confidence": verdict["confidence"],
        "description": (
            f"Left and right knee angles differ by {diff:.1f} deg on average "
            "across the video. Sustained left/right knee asymmetry is used "
            "here as an explainable proxy for uneven loading associated "
            "with knee valgus (inward knee collapse) patterns -- a true "
            "frontal-plane valgus angle would need a front-facing or "
            "multi-camera setup."
        ),
    }


def _check_poor_landing(movement_metrics: Dict, video_type: str) -> Optional[Dict]:
    """
    Only meaningful for 'jump' videos: a near-straight landing knee
    (little flexion) absorbs less shock than a bent-knee landing, a
    well-established landing-mechanics injury-risk pattern.
    """
    if video_type != "jump":
        return None

    landing_angle = movement_metrics.get("landing_knee_angle")
    if landing_angle is None:
        return None

    verdict = _level_and_confidence(
        landing_angle, POOR_LANDING_KNEE_ANGLE_LOW_DEG, POOR_LANDING_KNEE_ANGLE_HIGH_DEG
    )
    if verdict is None:
        return None

    return {
        "risk_type": "Poor Landing Technique",
        "affected_joint": "Knee",
        "risk_level": verdict["risk_level"],
        "confidence": verdict["confidence"],
        "description": (
            f"The landing knee angle reached {landing_angle:.1f} deg (close "
            "to fully straight). Landing with a straighter knee absorbs "
            "less impact and increases stress through the joint -- bending "
            "the knee more on landing spreads the impact over a longer "
            "time."
        ),
    }


def _check_shoulder_overextension(angle_summary: Dict) -> Optional[Dict]:
    """
    Checks whether either shoulder reaches a near-fully-extended peak
    angle. Repeated overextension during smashes is a known contributor
    to shoulder impingement / rotator-cuff strain.
    """
    left_stats = angle_summary.get("left_shoulder_angle") or {}
    right_stats = angle_summary.get("right_shoulder_angle") or {}
    left_max = left_stats.get("max")
    right_max = right_stats.get("max")
    candidates = [v for v in (left_max, right_max) if v is not None]
    if not candidates:
        return None

    peak = max(candidates)
    side = "left" if left_max == peak else "right"

    verdict = _level_and_confidence(
        peak, SHOULDER_OVEREXTENSION_LOW_DEG, SHOULDER_OVEREXTENSION_HIGH_DEG
    )
    if verdict is None:
        return None

    return {
        "risk_type": "Shoulder Overextension",
        "affected_joint": f"Shoulder ({side})",
        "risk_level": verdict["risk_level"],
        "confidence": verdict["confidence"],
        "description": (
            f"The {side} shoulder reached a peak angle of {peak:.1f} deg "
            "(near full extension) during the video. Repeated overextension "
            "of the hitting arm, especially during smashes, is associated "
            "with shoulder impingement and rotator-cuff strain over time."
        ),
    }


def _check_lunge_overreach(movement_metrics: Dict, video_type: str) -> Optional[Dict]:
    """
    Only meaningful for 'lunge' videos: how far the front knee travels
    past the front ankle at the deepest point of the lunge. Excessive
    forward knee travel increases shear stress on the knee joint.
    """
    if video_type != "lunge":
        return None

    offset = movement_metrics.get("knee_over_toe_offset")
    if offset is None:
        return None

    magnitude = abs(offset)
    verdict = _level_and_confidence(
        magnitude, LUNGE_KNEE_OVER_TOE_LOW, LUNGE_KNEE_OVER_TOE_HIGH
    )
    if verdict is None:
        return None

    return {
        "risk_type": "Lunge Overreach",
        "affected_joint": "Knee / Ankle",
        "risk_level": verdict["risk_level"],
        "confidence": verdict["confidence"],
        "description": (
            f"At the deepest point of the lunge, the front knee sat "
            f"{magnitude:.3f} (normalized image-width units) past the "
            "front ankle. Letting the knee travel too far past the toes "
            "increases shear stress on the knee joint -- keeping the shin "
            "closer to vertical reduces this stress."
        ),
    }

def _check_smash_technique_risk(
    movement_metrics: Dict, video_type: str
) -> Optional[Dict]:
    """
    Smash-specific technique-risk check.

    This uses project heuristics based on the smash metrics already
    calculated by smash_analysis.py. These are NOT clinical thresholds.

    The check is only applied to smash videos.
    """
    if video_type != "smash":
        return None

    elbow_angle = movement_metrics.get("max_elbow_extension_angle")
    rotation_range = movement_metrics.get("shoulder_rotation_range")

    candidates = []

    # Lower elbow extension can indicate incomplete extension during
    # the overhead swing.
    if elbow_angle is not None:
        elbow_deficit = max(
            0.0,
            SMASH_ELBOW_EXTENSION_FLOOR_DEG - float(elbow_angle),
        )

        if elbow_deficit > 0:
            verdict = _level_and_confidence(
                elbow_deficit,
                SMASH_ELBOW_EXTENSION_DEFICIT_LOW_DEG,
                SMASH_ELBOW_EXTENSION_DEFICIT_HIGH_DEG,
            )

            if verdict is not None:
                candidates.append(
                    (
                        elbow_deficit,
                        verdict,
                        "elbow",
                        float(elbow_angle),
                    )
                )

    # Limited shoulder rotation can indicate a mechanically restricted
    # or poorly coordinated overhead swing.
    if rotation_range is not None:
        rotation_deficit = max(
            0.0,
            SMASH_SHOULDER_ROTATION_FLOOR_DEG - float(rotation_range),
        )

        if rotation_deficit > 0:
            verdict = _level_and_confidence(
                rotation_deficit,
                SMASH_ROTATION_DEFICIT_LOW_DEG,
                SMASH_ROTATION_DEFICIT_HIGH_DEG,
            )

            if verdict is not None:
                candidates.append(
                    (
                        rotation_deficit,
                        verdict,
                        "shoulder",
                        float(rotation_range),
                    )
                )

    if not candidates:
        return None

    # Report the stronger of the detected smash-specific signals.
    _, verdict, signal, raw_value = max(
        candidates,
        key=lambda item: item[0],
    )

    if signal == "elbow":
        affected_joint = "Elbow / Shoulder"
        description = (
            f"The maximum elbow extension angle was {raw_value:.1f} deg, "
            "which is below the project's conservative smash-extension "
            "heuristic. Incomplete extension during an overhead swing "
            "may indicate inefficient technique and increased loading "
            "on surrounding joints."
        )
    else:
        affected_joint = "Shoulder"
        description = (
            f"The measured shoulder rotation range was {raw_value:.1f} deg, "
            "which is below the project's conservative smash-rotation "
            "heuristic. Limited rotation may indicate inefficient "
            "movement mechanics and increased loading on the hitting arm."
        )

    return {
        "risk_type": "Smash Technique Load Risk",
        "affected_joint": affected_joint,
        "risk_level": verdict["risk_level"],
        "confidence": verdict["confidence"],
        "description": description,
    }

def _check_asymmetrical_loading(angle_summary: Dict) -> Optional[Dict]:
    """
    Broader than knee valgus: looks at combined KNEE + HIP angle
    asymmetry between the two sides of the body across the whole video,
    as a proxy for consistently favoring/loading one leg more than the
    other -- a known contributor to overuse injuries.
    """
    knee_left = _get_avg(angle_summary, "left_knee_angle")
    knee_right = _get_avg(angle_summary, "right_knee_angle")
    hip_left = _get_avg(angle_summary, "left_hip_angle")
    hip_right = _get_avg(angle_summary, "right_hip_angle")

    diffs = []
    if knee_left is not None and knee_right is not None:
        diffs.append(abs(knee_left - knee_right))
    if hip_left is not None and hip_right is not None:
        diffs.append(abs(hip_left - hip_right))

    if not diffs:
        return None

    avg_diff = sum(diffs) / len(diffs)
    verdict = _level_and_confidence(
        avg_diff, HIP_KNEE_ASYMMETRY_LOW_DEG, HIP_KNEE_ASYMMETRY_HIGH_DEG
    )
    if verdict is None:
        return None

    return {
        "risk_type": "Asymmetrical Loading",
        "affected_joint": "Hip & Knee",
        "risk_level": verdict["risk_level"],
        "confidence": verdict["confidence"],
        "description": (
            f"Hip and knee angles differ between the left and right side "
            f"by {avg_diff:.1f} deg on average. Consistently favoring one "
            "side can lead to overuse injuries on the more-loaded leg over "
            "time -- balanced weight distribution reduces this risk."
        ),
    }


def detect_risks(
    angle_summary: Dict, movement_metrics: Dict, video_type: str
) -> List[Dict]:
    """
    Run every injury-risk check and return only the ones that triggered.

    Args:
        angle_summary: output of angle_calculator.aggregate_angles().
        movement_metrics: output of the video-type-specific analysis
            module (empty dict for 'general' videos).
        video_type: 'smash' / 'footwork' / 'jump' / 'lunge' / 'general'.

    Returns:
        A list of risk dicts, each shaped:
            {
                "risk_type": str, "affected_joint": str,
                "risk_level": "Low" | "Moderate" | "High",
                "confidence": float (0-100), "description": str,
            }
        Empty list if no risks were detected (or there wasn't enough
        data to check -- e.g. no person was detected in the video).
    """
    checks = [
        _check_knee_valgus(angle_summary),
        _check_poor_landing(movement_metrics, video_type),
        _check_shoulder_overextension(angle_summary),
        _check_lunge_overreach(movement_metrics, video_type),
        _check_smash_technique_risk(movement_metrics, video_type),
        _check_asymmetrical_loading(angle_summary),
    ]
    return [risk for risk in checks if risk is not None]


def calculate_overall_score(risks: List[Dict]) -> float:
    """
    Turn a list of detected risks into a single 0-100 "overall technique
    safety score" for the dashboard: 100 = no risks detected, reduced by
    a fixed penalty per risk depending on its severity (see
    _SCORE_PENALTY above).

    This is a simple, transparent, rule-based scale -- not a trained
    model -- so a viva panel can see exactly how any given score was
    produced.
    """
    score = 100.0
    for risk in risks:
        score -= _SCORE_PENALTY.get(risk.get("risk_level"), 10)
    return max(0.0, round(score, 1))
