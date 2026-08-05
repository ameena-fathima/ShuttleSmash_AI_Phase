"""
modules/recommendation_engine.py
-----------------------------------
Turns detected injury risks (from modules/injury_risk.py) into short,
plain-English coaching recommendations.

This is a simple lookup-table approach (risk_type -> recommendation
text) -- deliberately NOT machine learning, so every recommendation
shown to a player can be traced straight back to one specific rule,
which is easy to explain and defend in a viva.
"""

from typing import Dict, List

# risk_type -> base recommendation text. The same message is used
# regardless of risk_level; `priority` (below) is what actually scales
# with severity, since the advice itself doesn't change -- only how
# urgently it should be followed.
_RECOMMENDATION_TEXT = {
    "Knee Valgus Risk": (
        "Improve knee alignment during landing and lunges -- focus on "
        "keeping the knee tracking in line with the toes."
    ),
    "Poor Landing Technique": (
        "Increase knee flexion while landing to absorb impact more "
        "gradually and reduce joint stress."
    ),
    "Shoulder Overextension": (
        "Reduce arm extension during the smash and focus on generating "
        "power through trunk rotation instead."
    ),
    "Lunge Overreach": (
        "Shorten the lunge slightly so the front knee stays closer above "
        "the ankle, reducing shear stress on the knee."
    ),
    "Asymmetrical Loading": (
        "Work on balanced footwork drills so both legs share the load "
        "evenly instead of favoring one side."
    ),
}

# risk_type -> which broad category this recommendation falls under.
_RECOMMENDATION_CATEGORY = {
    "Knee Valgus Risk": "Injury Prevention",
    "Poor Landing Technique": "Technique",
    "Shoulder Overextension": "Technique",
    "Lunge Overreach": "Injury Prevention",
    "Asymmetrical Loading": "Fitness",
}

# risk_level -> recommendation priority. High risk = High priority to
# act on; Low risk still gets a Low-priority reminder rather than being
# silently dropped.
_PRIORITY_BY_RISK_LEVEL = {
    "High": "High",
    "Moderate": "Medium",
    "Low": "Low",
}


def generate_recommendations(risks: List[Dict]) -> List[Dict]:
    """
    Convert a list of detected risks into a list of recommendations.

    Args:
        risks: the list returned by modules/injury_risk.py's
            detect_risks(), each with at least "risk_type" and
            "risk_level".

    Returns:
        A list of dicts, each shaped:
            {"category": str, "recommendation": str, "priority": str}
        One recommendation per detected risk, in the same order. Risk
        types not found in the lookup table are skipped (rather than
        crashing), so this stays robust if injury_risk.py ever adds a
        new risk_type before this file is updated to match.
    """
    recommendations = []

    for risk in risks:
        risk_type = risk.get("risk_type")
        text = _RECOMMENDATION_TEXT.get(risk_type)
        if text is None:
            continue  # unknown risk type -- skip rather than guess

        recommendations.append(
            {
                "category": _RECOMMENDATION_CATEGORY.get(risk_type, "General"),
                "recommendation": text,
                "priority": _PRIORITY_BY_RISK_LEVEL.get(risk.get("risk_level"), "Medium"),
            }
        )

    return recommendations
