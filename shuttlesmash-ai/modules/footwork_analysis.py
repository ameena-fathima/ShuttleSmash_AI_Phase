"""
modules/footwork_analysis.py
------------------------------
Computes footwork metrics from a sequence of pose frames: lateral
movement range, step count (direction changes), average stance width,
and average movement speed.

These are descriptive, explainable metrics only -- no scoring, no
judgement, no injury inference (that's a later phase). They're exactly
the kind of features a Random Forest injury-risk model can eventually be
trained on, once labeled data exists.

Input shape: every function here takes `frame_records`, a chronological
list of dicts built by services/analysis_service.py, one per sampled
frame where a person was detected:
    {
        "frame_index": int,
        "timestamp": float,               # seconds from video start
        "landmarks": {"LEFT_HIP": {...}, "RIGHT_HIP": {...}, ...},
        "angles": {"left_knee_angle": 172.3, ...},
    }
"""

from typing import Dict, List, Optional


def _hip_center_x(frame: Dict) -> Optional[float]:
    """
    Average x-position of the left+right hip landmarks -- a simple,
    standard proxy for the player's horizontal position on court.
    """
    landmarks = frame["landmarks"]
    left_hip = landmarks.get("LEFT_HIP")
    right_hip = landmarks.get("RIGHT_HIP")
    if left_hip is None or right_hip is None:
        return None
    return (left_hip["x"] + right_hip["x"]) / 2


def analyze(frame_records: List[Dict], fps: float) -> Dict[str, Optional[float]]:
    """
    Compute footwork metrics across all sampled frames of a video.

    Args:
        frame_records: per-frame pose data, chronological order.
        fps: video's frames-per-second (used to convert frame-to-frame
            movement into a rough "per second" speed estimate).

    Returns:
        {
            "lateral_movement_range": float | None,  # 0-1 normalized width covered on screen
            "step_count": int,                         # direction changes in hip movement
            "avg_stance_width": float | None,          # normalized ankle-to-ankle distance
            "avg_movement_speed": float | None,        # normalized units / second
        }
    """
    empty_result = {
        "lateral_movement_range": None,
        "step_count": 0,
        "avg_stance_width": None,
        "avg_movement_speed": None,
    }
    if len(frame_records) < 2:
        return empty_result

    hip_x_positions = [_hip_center_x(f) for f in frame_records]
    hip_x_positions = [x for x in hip_x_positions if x is not None]
    if len(hip_x_positions) < 2:
        return empty_result

    lateral_movement_range = round(max(hip_x_positions) - min(hip_x_positions), 4)

    # Step count: count how many times the hip's horizontal movement
    # direction reverses. This is a simple, explainable proxy for
    # footwork steps / direction changes -- not a true biomechanical
    # step-detection algorithm, which would need higher frame rates.
    step_count = 0
    direction = 0
    for prev_x, curr_x in zip(hip_x_positions, hip_x_positions[1:]):
        delta = curr_x - prev_x
        if abs(delta) < 0.005:  # ignore tiny jitter/noise
            continue
        new_direction = 1 if delta > 0 else -1
        if direction != 0 and new_direction != direction:
            step_count += 1
        direction = new_direction

    # Stance width: horizontal distance between ankles, averaged across frames.
    stance_widths = []
    for f in frame_records:
        landmarks = f["landmarks"]
        left_ankle = landmarks.get("LEFT_ANKLE")
        right_ankle = landmarks.get("RIGHT_ANKLE")
        if left_ankle and right_ankle:
            stance_widths.append(abs(left_ankle["x"] - right_ankle["x"]))
    avg_stance_width = (
        round(sum(stance_widths) / len(stance_widths), 4) if stance_widths else None
    )

    # Average movement speed: total hip-center displacement over total time.
    total_displacement = sum(
        abs(b - a) for a, b in zip(hip_x_positions, hip_x_positions[1:])
    )
    duration = frame_records[-1]["timestamp"] - frame_records[0]["timestamp"]
    avg_movement_speed = round(total_displacement / duration, 4) if duration > 0 else None

    return {
        "lateral_movement_range": lateral_movement_range,
        "step_count": step_count,
        "avg_stance_width": avg_stance_width,
        "avg_movement_speed": avg_movement_speed,
    }
