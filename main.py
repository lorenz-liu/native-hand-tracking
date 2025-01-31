import cv2
import numpy as np

from native_hand_tracker import NativeHandTracker


def draw_info_on_frame(frame, landmarks, pinch_distance=None, pinch_center=None):
    """Draw additional information on the frame for visualization."""
    if not landmarks:
        return frame

    # Draw finger tip points with labels
    finger_tips = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}

    for finger_name, tip_idx in finger_tips.items():
        if len(landmarks) > tip_idx:
            x, y = landmarks[tip_idx][1], landmarks[tip_idx][2]
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            cv2.putText(
                frame,
                finger_name,
                (x - 20, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    # Draw pinch visualization if detected
    if pinch_distance is not None and pinch_center is not None:
        # Draw pinch center
        cv2.circle(frame, (pinch_center[0], pinch_center[1]), 8, (0, 255, 0), -1)

        # Display pinch distance
        cv2.putText(
            frame,
            f"Pinch: {pinch_distance:.1f}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Visual indicator for pinch state
        if pinch_distance < 40:  # Threshold for pinch detection
            cv2.putText(
                frame,
                "PINCH DETECTED!",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Draw line between thumb and index
            if len(landmarks) > 8:
                thumb_pos = (landmarks[4][1], landmarks[4][2])
                index_pos = (landmarks[8][1], landmarks[8][2])
                cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 2)

    return frame


def main():
    # Initialize camera and tracker
    cap = cv2.VideoCapture(0)
    tracker = NativeHandTracker()

    # Ensure camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Controls:")
    print("- Press 'q' to quit")
    print("- Try making a pinch gesture with your thumb and index finger")
    print("- Move your hand around to see tracking in action")

    while True:
        # Read frame from camera
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame")
            break

        # Flip frame horizontally for more intuitive interaction
        frame = cv2.flip(frame, 1)

        # Process the frame with hand tracker
        frame = tracker.process_frame(frame)

        # Get landmark positions
        landmarks = tracker.get_landmark_positions(frame)

        if landmarks:
            # Get pinch information
            pinch_distance, pinch_center = tracker.get_pinch_details(landmarks)

            # Draw additional visualization
            frame = draw_info_on_frame(frame, landmarks, pinch_distance, pinch_center)

            # Get and display hand rotation
            rotation = tracker.get_hand_rotation(landmarks)
            if rotation is not None:
                angle_degrees = np.degrees(rotation)
                cv2.putText(
                    frame,
                    f"Rotation: {angle_degrees:.1f} deg",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        # Add basic instructions
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Show the frame
        cv2.imshow("Hand Tracking Demo", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
