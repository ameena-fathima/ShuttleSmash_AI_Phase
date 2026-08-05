"""
modules/smash_analysis.py
---------------------------
Computes smash-technique metrics: how straight the hitting arm gets at
its highest point, how much the shoulders/trunk rotate through the
swing, an estimated contact frame, and swing speed around contact.

Same input shape as footwork_analysis.py -- see that file's docstring
for the `frame_records` layout.
"""

from typing import Dict, List, Optional


def _wrist_landmark(frame: Dict, side: str) -> Optional[Dict]:
    return frame["landmarks"].get(f"{side}_WRIST")


def analyze(frame_records: List[Dict], fps: float) -> Dict[str, Optional[float]]:
    """
    Compute smash metrics across all sampled frames of a video.

    Returns:
        {
            "max_elbow_extension_angle": float | None,  # straightest elbow angle reached (degrees)
            "shoulder_rotation_range": float | None,      # max-min shoulder angle across the swing (degrees)
            "contact_frame_index": int | None,             # frame index of the highest wrist point (contact proxy)
            "arm_swing_speed": float | None,                # normalized wrist displacement / second, around contact
        }
    """
    empty_result = {
        "max_elbow_extension_angle": None,
        "shoulder_rotation_range": None,
        "contact_frame_index": None,
        "arm_swing_speed": None,
    }
    if not frame_records:
        return empty_result

    elbow_angles = []
    shoulder_angles = []
    for f in frame_records:
        for side in ("left", "right"):
            elbow = f["angles"].get(f"{side}_elbow_angle")
            shoulder = f["angles"].get(f"{side}_shoulder_angle")
            if elbow is not None:
                elbow_angles.append(elbow)
            if shoulder is not None:
                shoulder_angles.append(shoulder)

    max_elbow_extension_angle = round(max(elbow_angles), 2) if elbow_angles else None
    shoulder_rotation_range = (
        round(max(shoulder_angles) - min(shoulder_angles), 2) if shoulder_angles else None
    )

    # Contact point proxy: the frame where either wrist reaches its
    # highest point on screen (smallest y value, since image y grows
    # downward -- "highest" = smallest y).
    contact_frame_index = None
    best_y = None
    for f in frame_records:
        for side in ("LEFT", "RIGHT"):
            wrist = _wrist_landmark(f, side)
            if wrist is None:
                continue
            if best_y is None or wrist["y"] < best_y:
                best_y = wrist["y"]
                contact_frame_index = f["frame_index"]

    # Arm swing speed: wrist displacement across the two frames straddling
    # contact, divided by the time between them -- a rough peak-speed proxy.
    arm_swing_speed = None
    if contact_frame_index is not None and len(frame_records) >= 3:
        contact_idx = next(
            (i for i, f in enumerate(frame_records) if f["frame_index"] == contact_frame_index),
            None,
        )
        if contact_idx is not None and 0 < contact_idx < len(frame_records) - 1:
            prev_f, next_f = frame_records[contact_idx - 1], frame_records[contact_idx + 1]
            prev_wrist = _wrist_landmark(prev_f, "RIGHT") or _wrist_landmark(prev_f, "LEFT")
            next_wrist = _wrist_landmark(next_f, "RIGHT") or _wrist_landmark(next_f, "LEFT")
            time_delta = next_f["timestamp"] - prev_f["timestamp"]
            if prev_wrist and next_wrist and time_delta > 0:
                displacement = (
                    (next_wrist["x"] - prev_wrist["x"]) ** 2
                    + (next_wrist["y"] - prev_wrist["y"]) ** 2
                ) ** 0.5
                arm_swing_speed = round(displacement / time_delta, 4)

    return {
        "max_elbow_extension_angle": max_elbow_extension_angle,
        "shoulder_rotation_range": shoulder_rotation_range,
        "contact_frame_index": contact_frame_index,
        "arm_swing_speed": arm_swing_speed,
    }
