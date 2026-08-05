"""
modules/pose_estimation.py
----------------------------
A thin, beginner-friendly wrapper around MediaPipe's Pose solution.

Why wrap it at all? Two reasons:
  1. The rest of the codebase never has to import `mediapipe` directly --
     everything goes through this one class. If MediaPipe's API ever
     changes, only this file needs updating.
  2. It bundles the "initialize once, reuse many times" pattern that
     MediaPipe expects. Creating a new Pose model for every single frame
     is a common beginner mistake that makes processing extremely slow --
     PoseEstimator is designed to be created ONCE per video and reused
     across all of that video's sampled frames.

Phase 2 scope: detect landmarks and draw the skeleton overlay only.
No badminton-specific scoring happens here -- that's added in Phase 3
(modules/angle_calculator.py and friends), which will consume the
landmark dictionaries this class returns.
"""

from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """
    Detects human body pose (33 landmarks) in video frames using
    MediaPipe, and can draw a skeleton overlay for visual preview.

    Typical usage (as a context manager, which guarantees MediaPipe's
    resources are released even if an error occurs partway through):

        with PoseEstimator() as estimator:
            for frame_index, frame in sampled_frames:
                results = estimator.detect(frame)
                landmarks = estimator.get_landmarks(results)
                if landmarks is not None:
                    overlay = estimator.draw_skeleton(frame, results)
    """

    # Human-readable names for MediaPipe's 33 body landmarks, in index
    # order (0 = NOSE, 1 = LEFT_EYE_INNER, ... 32 = RIGHT_FOOT_INDEX).
    LANDMARK_NAMES: List[str] = [lm.name for lm in mp.solutions.pose.PoseLandmark]

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Create and initialize the MediaPipe Pose model.

        Args:
            static_image_mode: False treats frames as a continuous video
                stream (MediaPipe tracks landmarks between frames, which
                is faster and smoother). True treats every frame as an
                unrelated still image (more accurate for one-off photos,
                but slower for video). Video upload -> use False.
            model_complexity: 0 (fastest, least accurate) to 2 (slowest,
                most accurate). 1 is a good balance for a college project
                running on a normal laptop.
            min_detection_confidence: minimum confidence (0-1) required to
                consider a person "detected" in a frame.
            min_tracking_confidence: minimum confidence (0-1) required to
                keep tracking landmarks between frames (video mode only).

        Raises:
            RuntimeError: if the installed `mediapipe` package doesn't
                expose the legacy Solutions API this class relies on
                (mp.solutions.pose). This happens with mediapipe>=0.10.22
                on some platforms, which dropped that API from the wheel.
                See the comment above the mediapipe line in
                requirements.txt for the pinned working version.
        """
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Your installed `mediapipe` package does not include the "
                "legacy Solutions API (mp.solutions) that PoseEstimator "
                "relies on. This happens with mediapipe 0.10.22+ on some "
                "platforms. Fix it by running:\n"
                "    pip install mediapipe==0.10.21\n"
                "(see requirements.txt for details)"
            )

        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        # The model itself is created ONCE here, not per frame.
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> Any:
        """
        Run pose detection on a single video frame.

        Args:
            frame: a BGR image as a numpy array (OpenCV's default format,
                e.g. straight from cv2.VideoCapture.read()).

        Returns:
            MediaPipe's raw results object. `results.pose_landmarks` will
            be None if no person was detected in the frame -- use
            get_landmarks() to safely handle that case.
        """
        # MediaPipe expects RGB images; OpenCV gives us BGR by default.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # perf hint recommended by MediaPipe docs
        results = self._pose.process(rgb_frame)
        return results

    def get_landmarks(self, results: Any) -> Optional[List[Dict[str, float]]]:
        """
        Convert MediaPipe's raw results into a plain list of dictionaries
        that the rest of the app can use without knowing anything about
        MediaPipe's internal types.

        Args:
            results: the object returned by detect().

        Returns:
            None if no person was detected in the frame (graceful
            handling of empty frames). Otherwise, a list of 33 dicts:
                {"id": int, "name": str, "x": float, "y": float,
                 "z": float, "visibility": float}
            x/y are normalized image coordinates (0.0-1.0); z is relative
            depth; visibility is the model's confidence (0.0-1.0) that the
            landmark is actually visible (not occluded/off-screen).
        """
        if results is None or results.pose_landmarks is None:
            return None

        landmarks = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append(
                {
                    "id": idx,
                    "name": self.LANDMARK_NAMES[idx],
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }
            )
        return landmarks

    def draw_skeleton(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw the pose skeleton overlay on a COPY of the given frame.

        Args:
            frame: the original BGR frame the results came from.
            results: the object returned by detect() for this same frame.

        Returns:
            A new image (original frame is never modified in place). If
            no person was detected, the original frame is returned
            unchanged, so callers can always safely use the return value
            for a preview image -- even for frames with nobody in them.
        """
        annotated_frame = frame.copy()

        if results is None or results.pose_landmarks is None:
            return annotated_frame

        self._mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        return annotated_frame

    def close(self) -> None:
        """
        Release MediaPipe's internal resources.
        Always call this when finished (or use PoseEstimator as a context
        manager with `with`, which calls this automatically).
        """
        self._pose.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
