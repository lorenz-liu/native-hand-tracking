import math
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class NativeHandTracker:
    """
    A class for tracking hands using the device's native camera.
    Provides hand landmark detection and gesture analysis capabilities.
    """

    def __init__(
        self,
        static_mode: bool = False,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ):
        """
        Initialize the hand tracker.

        Args:
            static_mode: Whether to treat each frame independently for detection
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for landmark tracking
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Store the latest results
        self.results = None

        # Define landmark indices for common fingers
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20

    def process_frame(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Process a single frame and detect hands.

        Args:
            frame: Input image frame
            draw: Whether to draw landmarks on the frame

        Returns:
            Processed frame with optional landmark drawings
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        self.results = self.hands.process(frame_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(
                        color=(255, 255, 255), thickness=2, circle_radius=2
                    ),  # White dots
                    self.mp_draw.DrawingSpec(
                        color=(255, 255, 255), thickness=2
                    ),  # White connections
                )

        return frame

    def get_landmark_positions(
        self, frame: np.ndarray, hand_idx: int = 0
    ) -> List[List[int]]:
        """
        Get positions of all landmarks for a specific hand.

        Args:
            frame: Input image frame
            hand_idx: Index of the hand to get landmarks for

        Returns:
            List of [id, x, y] coordinates for each landmark
        """
        landmark_list = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_idx:
                hand = self.results.multi_hand_landmarks[hand_idx]
                height, width, _ = frame.shape

                for idx, landmark in enumerate(hand.landmark):
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([idx, cx, cy])

        return landmark_list

    def get_pinch_details(
        self, landmark_list: List[List[int]]
    ) -> Tuple[Optional[float], Optional[List[int]]]:
        """
        Calculate pinch distance and center point between thumb and index finger.

        Args:
            landmark_list: List of landmark positions

        Returns:
            Tuple of (pinch_distance, pinch_center)
        """
        if len(landmark_list) <= max(self.THUMB_TIP, self.INDEX_TIP):
            return None, None

        thumb_tip = landmark_list[self.THUMB_TIP]
        index_tip = landmark_list[self.INDEX_TIP]

        # Calculate distance
        distance = math.sqrt(
            (thumb_tip[1] - index_tip[1]) ** 2 + (thumb_tip[2] - index_tip[2]) ** 2
        )

        # Calculate center point
        center = [
            (thumb_tip[1] + index_tip[1]) // 2,
            (thumb_tip[2] + index_tip[2]) // 2,
        ]

        return distance, center

    def get_hand_rotation(self, landmark_list: List[List[int]]) -> Optional[float]:
        """
        Calculate the rotation angle of the hand based on thumb and index positions.

        Args:
            landmark_list: List of landmark positions

        Returns:
            Angle in radians, or None if landmarks not found
        """
        if len(landmark_list) <= max(self.THUMB_TIP, self.INDEX_TIP):
            return None

        thumb_tip = landmark_list[self.THUMB_TIP]
        index_tip = landmark_list[self.INDEX_TIP]

        angle = math.atan2(index_tip[2] - thumb_tip[2], index_tip[1] - thumb_tip[1])

        return angle

    def is_pinching(
        self, landmark_list: List[List[int]], threshold: float = 40.0
    ) -> bool:
        """
        Determine if a pinching gesture is being made.

        Args:
            landmark_list: List of landmark positions
            threshold: Maximum distance threshold for pinch detection

        Returns:
            True if pinching gesture detected, False otherwise
        """
        distance, _ = self.get_pinch_details(landmark_list)
        return distance is not None and distance < threshold

    def release(self):
        """
        Release resources used by the hand tracker.
        """
        self.hands.close()
