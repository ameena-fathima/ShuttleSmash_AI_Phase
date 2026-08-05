"""
modules/jump_analysis.py
--------------------------
Computes jump metrics: estimated jump height (relative to body height,
since a single uncalibrated camera can't measure real centimeters),
landing knee flexion, and flight time.

Same input shape as footwork_analysis.py -- see that file's docstring
for the `frame_records` layout.
"""

from typing import Dict, List, Optional


def _hip_y(frame: Dict) -> Optional[float]:
    """Average y-position of both hips -- lower on screen (bigger y) = closer to the ground."""
    landmarks = frame["landmarks"]
    left_hip = landmarks.get("LEFT_HIP")
    right_hip = landmarks.get("RIGHT_HIP")
    if left_hip is None or right_hip is None:
        return None
    return (left_hip["y"] + right_hip["y"]) / 2


def _body_height_reference(frame: Dict) -> Optional[float]:
    """
    Approximate the player's on-screen body height (shoulder-to-ankle
    distance), used to convert vertical hip movement from normalized
    image units into a "percentage of body height" jump estimate.

    A true centimeter measurement would require a calibrated camera
    (known distance/angle/focal length), which is out of scope for a
    single uncalibrated video. Expressing jump height relative to body
    height is a standard, defensible substitute in sports-video analysis
    without dedicated calibration hardware.
    """
    landmarks = frame["landmarks"]
    shoulder = landmarks.get("LEFT_SHOULDER") or landmarks.get("RIGHT_SHOULDER")
    ankle = landmarks.get("LEFT_ANKLE") or landmarks.get("RIGHT_ANKLE")
    if shoulder is None or ankle is None:
        return None
    return abs(ankle["y"] - shoulder["y"])


def analyze(frame_records: List[Dict], fps: float) -> Dict[str, Optional[float]]:
    """
    Compute jump metrics across all sampled frames of a video.

    Returns:
        {
            "jump_height_percent_body": float | None,  # peak hip rise, as % of body height
            "landing_knee_angle": float | None,          # more-bent knee angle at landing (degrees)
            "flight_time_seconds": float | None,         # time from takeoff frame to landing frame
            "peak_frame_index": int | None,              # frame index of the highest point of the jump
        }
    """
    empty_result = {
        "jump_height_percent_body": None,
        "landing_knee_angle": None,
        "flight_time_seconds": None,
        "peak_frame_index": None,
    }
    if len(frame_records) < 3:
        return empty_result

    hip_ys = [(_hip_y(f), f) for f in frame_records]
    hip_ys = [(y, f) for y, f in hip_ys if y is not None]
    if len(hip_ys) < 3:
        return empty_result

    # Baseline standing hip height = the LOWEST point on screen the hip
    # reaches (largest y value -- closest to the ground/floor).
    baseline_y = max(y for y, _ in hip_ys)
    # Peak of the jump = the frame with the SMALLEST hip y (highest point).
    peak_y, peak_frame = min(hip_ys, key=lambda item: item[0])

    body_height = _body_height_reference(peak_frame) or _body_height_reference(frame_records[0])
    jump_height_percent_body = None
    if body_height and body_height > 0:
        jump_height_percent_body = round(((baseline_y - peak_y) / body_height) * 100, 2)

    # Landing frame: the first frame AFTER the peak where the hip has
    # dropped back to within 20% of its original rise -- i.e. "mostly
    # back down".
    peak_index = frame_records.index(peak_frame)
    rise = baseline_y - peak_y
    landing_frame = None
    if rise > 0:
        for f in frame_records[peak_index:]:
            y = _hip_y(f)
            if y is not None and (baseline_y - y) < rise * 0.2:
                landing_frame = f
                break

    landing_knee_angle = None
    if landing_frame is not None:
        left_knee = landing_frame["angles"].get("left_knee_angle")
        right_knee = landing_frame["angles"].get("right_knee_angle")
        knee_values = [v for v in (left_knee, right_knee) if v is not None]
        if knee_values:
            # The more-bent (smaller angle) knee is treated as the main
            # landing leg for a single-leg or uneven landing.
            landing_knee_angle = round(min(knee_values), 2)

    flight_time_seconds = None
    if landing_frame is not None:
        flight_time_seconds = round(
            landing_frame["timestamp"] - frame_records[0]["timestamp"], 3
        )

    return {
        "jump_height_percent_body": jump_height_percent_body,
        "landing_knee_angle": landing_knee_angle,
        "flight_time_seconds": flight_time_seconds,
        "peak_frame_index": peak_frame["frame_index"],
    }
