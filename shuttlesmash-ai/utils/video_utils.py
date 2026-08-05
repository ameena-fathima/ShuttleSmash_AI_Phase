"""
utils/video_utils.py
---------------------
OpenCV helper functions for reading badminton video files: opening the
file, reading its metadata (fps/duration/resolution/frame count), and
pulling out a manageable subset of frames for the pose-estimation
pipeline.

Running MediaPipe on every single frame of a video would be slow and
mostly redundant, since adjacent frames in a 30fps clip barely differ --
that's why sample_frames() exists.
"""

import os
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np


def open_video(video_path: str) -> cv2.VideoCapture:
    """
    Open a video file with OpenCV.

    Args:
        video_path: absolute or relative path to a video file on disk.

    Returns:
        An opened cv2.VideoCapture ready to be read from. The caller is
        responsible for calling .release() on it when done (the other
        functions in this module do this automatically).

    Raises:
        FileNotFoundError: if the path doesn't exist on disk.
        IOError: if OpenCV can't open/decode the file (corrupted file or
            unsupported codec).
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"OpenCV could not open video file: {video_path}")
    return capture


def get_video_information(video_path: str) -> Dict[str, object]:
    """
    Read basic metadata from a video file (fast -- reads header info only,
    does not decode every frame).

    Args:
        video_path: path to the video file on disk.

    Returns:
        A dict with keys:
            fps (float), frame_count (int), duration_seconds (float),
            width (int), height (int), resolution (str, e.g. "1280x720")
    """
    capture = open_video(video_path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_seconds = round(frame_count / fps, 2) if fps > 0 else 0.0

        return {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
        }
    finally:
        # Always release the capture (frees the underlying video file
        # handle), even if reading a property above somehow raises.
        capture.release()


def extract_frames(video_path: str) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Lazily read every frame of a video, one at a time.

    This is a generator function (it uses `yield`), so frames are decoded
    one-by-one as they're consumed instead of all being loaded into memory
    at once -- important for longer videos, which could otherwise use a
    huge amount of RAM if every frame were collected into a list upfront.

    Args:
        video_path: path to the video file on disk.

    Yields:
        (frame_index, frame) tuples, where frame_index is the frame's
        position in the video (starting at 0) and frame is a BGR numpy
        array (OpenCV's default color format).
    """
    capture = open_video(video_path)
    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break  # end of video (or unreadable frame)
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def sample_frames(
    video_path: str,
    sample_rate: int = 5,
    max_frames: Optional[int] = None,
) -> List[Tuple[int, np.ndarray]]:
    """
    Return every Nth frame of a video instead of every single frame.

    This is the main speed optimization for the pose-estimation pipeline:
    a 10-second clip at 30fps has 300 frames, but sampling every 5th frame
    reduces that to 60 frames to run MediaPipe on, with negligible loss of
    information since consecutive frames are nearly identical.

    Args:
        video_path: path to the video file on disk.
        sample_rate: keep 1 out of every `sample_rate` frames
            (e.g. 5 keeps frame 0, 5, 10, 15, ...). Must be >= 1.
        max_frames: optional safety cap on how many sampled frames to
            return, in case a very long video is uploaded. None = no cap.

    Returns:
        A list of (frame_index, frame) tuples. frame_index refers to the
        position in the ORIGINAL (un-sampled) video, which is useful later
        for converting a frame back into a timestamp (frame_index / fps).
    """
    if sample_rate < 1:
        raise ValueError("sample_rate must be at least 1.")

    sampled: List[Tuple[int, np.ndarray]] = []
    for frame_index, frame in extract_frames(video_path):
        if frame_index % sample_rate == 0:
            sampled.append((frame_index, frame))
            if max_frames is not None and len(sampled) >= max_frames:
                break
    return sampled
