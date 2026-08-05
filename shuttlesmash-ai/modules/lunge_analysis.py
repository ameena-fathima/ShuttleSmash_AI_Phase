"""
modules/lunge_analysis.py
---------------------------
Computes lunge metrics: how deep the lunge goes (knee flexion), how far
the front knee travels past the ankle at the deepest point, and how much
the torso leans forward.

Same input shape as footwork_analysis.py -- see that file's docstring
for the `frame_records` layout.
"""

import math
from typing import Dict, List, Optional


def analyze(frame_records: List[Dict], fps: float) -> Dict[str, Optional[float]]:
    """
    Compute lunge metrics across all sampled frames of a video.

    Returns:
        {
            "max_knee_flexion_angle": float | None,  # smallest knee angle reached = deepest lunge (degrees)
            "knee_over_toe_offset": float | None,       # front knee x - front ankle x at the deepest frame (normalized)
            "torso_lean_angle": float | None,            # shoulder-hip line's angle from vertical, degrees, at the deepest frame
        }
    """
    empty_result = {
        "max_knee_flexion_angle": None,
        "knee_over_toe_offset": None,
        "torso_lean_angle": None,
    }
    if not frame_records:
        return empty_result

    # Find the frame + leg with the deepest lunge = the smallest
    # (most bent) knee angle seen anywhere in the video.
    deepest_frame = None
    deepest_angle = None
    deepest_side = None
    for f in frame_records:
        for side in ("left", "right"):
            angle = f["angles"].get(f"{side}_knee_angle")
            if angle is None:
                continue
            if deepest_angle is None or angle < deepest_angle:
                deepest_angle = angle
                deepest_frame = f
                deepest_side = side

    if deepest_frame is None:
        return empty_result

    landmarks = deepest_frame["landmarks"]
    side_upper = deepest_side.upper()
    knee = landmarks.get(f"{side_upper}_KNEE")
    ankle = landmarks.get(f"{side_upper}_ANKLE")
    knee_over_toe_offset = None
    if knee and ankle:
        # Positive = knee is further forward (toward the direction of
        # travel) than the ankle at the deepest point of the lunge.
        knee_over_toe_offset = round(knee["x"] - ankle["x"], 4)

    shoulder = landmarks.get(f"{side_upper}_SHOULDER")
    hip = landmarks.get(f"{side_upper}_HIP")
    torso_lean_angle = None
    if shoulder and hip:
        dx = shoulder["x"] - hip["x"]
        dy = shoulder["y"] - hip["y"]
        # Angle from vertical: 0 = perfectly upright torso, larger = more forward lean.
        torso_lean_angle = round(math.degrees(math.atan2(abs(dx), abs(dy))), 2)

    return {
        "max_knee_flexion_angle": round(deepest_angle, 2),
        "knee_over_toe_offset": knee_over_toe_offset,
        "torso_lean_angle": torso_lean_angle,
    }
