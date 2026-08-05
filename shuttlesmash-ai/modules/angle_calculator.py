"""
modules/angle_calculator.py
------------------------------
Pure geometry: turns MediaPipe body landmarks into joint angles.

No MediaPipe/Flask/DB imports here -- this module only knows about plain
landmark dictionaries (the same {"name":..., "x":..., "y":..., "z":...,
"visibility":...} shape returned by PoseEstimator.get_landmarks()), so it
can be unit-tested and reused completely independently of the rest of
the app.

All angles are in DEGREES, measured at the middle landmark of three
points using the standard "angle between two vectors" formula:
    angle = arccos( (BA . BC) / (|BA| * |BC|) )
Only x/y (the image plane) are used -- a full 3D angle would need a
calibrated multi-camera rig to be reliable, which is out of scope for a
single uncalibrated video. This 2D approach is the standard, defensible
choice for single-camera sports-video analysis.
"""

import math
from typing import Dict, List, Optional

# Minimum visibility score (0-1) a landmark must have before we trust it
# enough to use in an angle calculation. Below this, MediaPipe itself is
# unsure the landmark is actually visible, so using it would produce a
# misleading angle.
MIN_VISIBILITY = 0.5

# Every joint angle this module calculates, defined as
# (point_a_suffix, vertex_suffix, point_c_suffix). The angle is always
# measured AT the vertex landmark. Combined with a side ("LEFT"/"RIGHT")
# this maps directly onto MediaPipe's landmark names, e.g.
# "elbow" + "LEFT" -> LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST.
JOINT_DEFINITIONS = {
    "elbow": ("SHOULDER", "ELBOW", "WRIST"),
    "shoulder": ("ELBOW", "SHOULDER", "HIP"),
    "hip": ("SHOULDER", "HIP", "KNEE"),
    "knee": ("HIP", "KNEE", "ANKLE"),
    "ankle": ("KNEE", "ANKLE", "FOOT_INDEX"),
}

SIDES = ("LEFT", "RIGHT")


def _landmarks_by_name(landmarks: List[Dict]) -> Dict[str, Dict]:
    """Index a landmark list by name for O(1) lookup, e.g. lookup['LEFT_KNEE']."""
    return {lm["name"]: lm for lm in landmarks}


def calculate_angle(
    point_a: Dict[str, float], vertex: Dict[str, float], point_c: Dict[str, float]
) -> float:
    """
    Calculate the angle (in degrees) at `vertex`, between the rays
    vertex->point_a and vertex->point_c.

    Args:
        point_a, vertex, point_c: landmark dicts with at least "x", "y"
            (normalized 0-1 image coordinates).

    Returns:
        Angle in degrees, always between 0 and 180.
    """
    vector_ba = (point_a["x"] - vertex["x"], point_a["y"] - vertex["y"])
    vector_bc = (point_c["x"] - vertex["x"], point_c["y"] - vertex["y"])

    dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]
    magnitude_ba = math.hypot(*vector_ba)
    magnitude_bc = math.hypot(*vector_bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0  # degenerate case: two landmarks sit on the same point

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    # Clamp to [-1, 1] to guard against floating-point rounding errors
    # that could otherwise make math.acos() raise a ValueError.
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))


def _joint_visible(*landmarks: Dict) -> bool:
    """True only if every given landmark meets the minimum visibility bar."""
    return all(lm.get("visibility", 0) >= MIN_VISIBILITY for lm in landmarks)


def calculate_all_angles(landmarks: List[Dict]) -> Dict[str, Optional[float]]:
    """
    Calculate every joint angle (elbow/shoulder/hip/knee/ankle, both
    sides) for a single frame of pose landmarks.

    Args:
        landmarks: the list of 33 landmark dicts from
            PoseEstimator.get_landmarks() for one frame.

    Returns:
        A flat dict, always the same 10 keys regardless of input, e.g.:
            {
                "left_elbow_angle": 142.3, "right_elbow_angle": 138.9,
                "left_shoulder_angle": 88.1, "right_shoulder_angle": 91.4,
                "left_hip_angle": 168.2, "right_hip_angle": 165.9,
                "left_knee_angle": 176.0, "right_knee_angle": 140.2,
                "left_ankle_angle": 95.3, "right_ankle_angle": None,
            }
        A value is None if the needed landmarks weren't visible enough
        to trust (rather than the key being missing entirely) -- keeping
        every returned dict the same shape matters later for building
        consistent CSV columns for ML training.
    """
    by_name = _landmarks_by_name(landmarks)
    angles: Dict[str, Optional[float]] = {}

    for joint_name, (a_suffix, vertex_suffix, c_suffix) in JOINT_DEFINITIONS.items():
        for side in SIDES:
            key = f"{side.lower()}_{joint_name}_angle"
            point_a = by_name.get(f"{side}_{a_suffix}")
            vertex = by_name.get(f"{side}_{vertex_suffix}")
            point_c = by_name.get(f"{side}_{c_suffix}")

            if point_a is None or vertex is None or point_c is None:
                angles[key] = None
                continue

            if not _joint_visible(point_a, vertex, point_c):
                angles[key] = None
                continue

            angles[key] = round(calculate_angle(point_a, vertex, point_c), 2)

    return angles


def aggregate_angles(
    per_frame_angles: List[Dict[str, Optional[float]]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Summarize a list of per-frame angle dicts (one per sampled frame that
    had a detected person) into average/min/max per joint angle.

    This summary is what actually gets stored in the database and used
    as ML features -- raw per-frame angles stay in memory only during
    processing (see the architecture doc's rule against storing
    thousands of per-frame rows in SQLite).

    Args:
        per_frame_angles: list of dicts, each shaped like the output of
            calculate_all_angles() for one frame.

    Returns:
        {
            "left_elbow_angle": {"avg": 140.2, "min": 110.5, "max": 178.9},
            ...
        }
        A joint with no valid readings across every frame still appears,
        as {"avg": None, "min": None, "max": None} -- again, for a
        consistent shape.
    """
    if not per_frame_angles:
        return {}

    summary: Dict[str, Dict[str, Optional[float]]] = {}
    all_keys = per_frame_angles[0].keys()

    for key in all_keys:
        values = [frame[key] for frame in per_frame_angles if frame.get(key) is not None]
        if values:
            summary[key] = {
                "avg": round(sum(values) / len(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            }
        else:
            summary[key] = {"avg": None, "min": None, "max": None}

    return summary
