"""
Project 6: Gesture-Controlled Presentation System
==================================================
Control PowerPoint/PDF presentations using hand gestures - perfect for
touchless presenting.

Key Concepts:
- Hand landmark detection (21 points)
- Gesture recognition (swipe, point, thumbs up)
- Screen interaction automation
- Gesture-to-action mapping
- Smooth gesture debouncing

Official OpenCV References:
- Video Analysis: https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import time

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not installed. Install with: pip install mediapipe")

# Try to import pyautogui for screen control
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("pyautogui not installed. Screen control disabled.")
    print("Install with: pip install pyautogui")


class HandDetector:
    """
    Hand detection and landmark extraction using MediaPipe.
    """

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for hand detection")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20

    def detect(self, image):
        """
        Detect hands and extract landmarks.

        Args:
            image: BGR image

        Returns:
            List of hand landmarks (normalized coordinates)
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z
                    })
                hands.append(landmarks)

        return hands

    def draw_landmarks(self, image, landmarks):
        """Draw hand landmarks on image."""
        if not landmarks:
            return image

        h, w, _ = image.shape

        for hand in landmarks:
            # Convert to pixel coordinates
            points = [(int(lm['x'] * w), int(lm['y'] * h)) for lm in hand]

            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]

            for start, end in connections:
                cv2.line(image, points[start], points[end], (0, 255, 0), 2)

            # Draw points
            for i, point in enumerate(points):
                color = (255, 0, 0) if i in [4, 8, 12, 16, 20] else (0, 0, 255)
                cv2.circle(image, point, 5, color, -1)

        return image

    def get_finger_states(self, landmarks):
        """
        Determine which fingers are extended.

        Returns:
            List of booleans [thumb, index, middle, ring, pinky]
        """
        if not landmarks or len(landmarks) < 21:
            return [False] * 5

        lm = landmarks

        # Finger tip and pip (second joint) indices
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]

        fingers = []

        # Thumb (compare x instead of y)
        if lm[4]['x'] > lm[3]['x']:  # Right hand
            fingers.append(lm[4]['x'] > lm[3]['x'])
        else:  # Left hand
            fingers.append(lm[4]['x'] < lm[3]['x'])

        # Other fingers (tip above pip = extended)
        for tip, pip in zip(tips[1:], pips[1:]):
            fingers.append(lm[tip]['y'] < lm[pip]['y'])

        return fingers


class GestureRecognizer:
    """
    Recognize gestures from hand landmarks.
    """

    def __init__(self):
        self.position_history = deque(maxlen=10)
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # seconds

    def recognize(self, landmarks, finger_states):
        """
        Recognize gesture from landmarks and finger states.

        Returns:
            Gesture name or None
        """
        if not landmarks:
            return None

        # Current index finger position
        index_pos = (landmarks[8]['x'], landmarks[8]['y'])
        self.position_history.append(index_pos)

        gesture = None

        # Pointing (only index extended)
        if finger_states == [False, True, False, False, False]:
            gesture = 'pointing'

        # Open palm (all fingers extended)
        elif all(finger_states):
            gesture = 'open_palm'

        # Fist (no fingers extended)
        elif not any(finger_states):
            gesture = 'fist'

        # Peace sign (index and middle extended)
        elif finger_states == [False, True, True, False, False]:
            gesture = 'peace'

        # Thumbs up
        elif finger_states == [True, False, False, False, False]:
            gesture = 'thumbs_up'

        # Swipe detection
        swipe = self._detect_swipe()
        if swipe:
            gesture = swipe

        return gesture

    def _detect_swipe(self, threshold=0.15):
        """Detect swipe gestures from position history."""
        if len(self.position_history) < 5:
            return None

        # Get movement
        start = self.position_history[0]
        end = self.position_history[-1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Check if movement is significant
        if abs(dx) > threshold:
            self.position_history.clear()
            return 'swipe_right' if dx > 0 else 'swipe_left'

        if abs(dy) > threshold:
            self.position_history.clear()
            return 'swipe_down' if dy > 0 else 'swipe_up'

        return None


class PresentationController:
    """
    Control presentations using detected gestures.
    """

    def __init__(self):
        self.last_action_time = 0
        self.action_cooldown = 0.8  # seconds

        # Gesture to action mapping
        self.gesture_actions = {
            'swipe_left': self.next_slide,
            'swipe_right': self.prev_slide,
            'open_palm': self.pause,
            'fist': self.resume,
            'thumbs_up': self.volume_up,
            'peace': self.volume_down,
        }

    def handle_gesture(self, gesture):
        """Handle detected gesture."""
        if gesture is None:
            return None

        # Check cooldown
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return None

        if gesture in self.gesture_actions:
            action = self.gesture_actions[gesture]
            result = action()
            if result:
                self.last_action_time = current_time
            return result

        return None

    def next_slide(self):
        """Go to next slide."""
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('right')
        print("Action: Next slide")
        return 'next_slide'

    def prev_slide(self):
        """Go to previous slide."""
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('left')
        print("Action: Previous slide")
        return 'prev_slide'

    def pause(self):
        """Pause presentation."""
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('b')  # Black screen in PowerPoint
        print("Action: Pause")
        return 'pause'

    def resume(self):
        """Resume presentation."""
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('b')  # Toggle black screen
        print("Action: Resume")
        return 'resume'

    def volume_up(self):
        """Increase volume."""
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('volumeup')
        print("Action: Volume up")
        return 'volume_up'

    def volume_down(self):
        """Decrease volume."""
        if PYAUTOGUI_AVAILABLE:
            pyautogui.press('volumedown')
        print("Action: Volume down")
        return 'volume_down'


def run_gesture_control():
    """Run gesture-controlled presentation system."""
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe required. Install with: pip install mediapipe")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    detector = HandDetector()
    recognizer = GestureRecognizer()
    controller = PresentationController()

    print("Gesture Control System Started")
    print("=" * 40)
    print("Gestures:")
    print("  Swipe Left  -> Next slide")
    print("  Swipe Right -> Previous slide")
    print("  Open Palm   -> Pause/Black screen")
    print("  Fist        -> Resume")
    print("  Thumbs Up   -> Volume up")
    print("  Peace Sign  -> Volume down")
    print()
    print("Press Q to quit")

    current_gesture = None
    last_action = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Detect hands
        hands = detector.detect(frame)

        if hands:
            # Get first hand
            landmarks = hands[0]

            # Draw landmarks
            display = detector.draw_landmarks(display, hands)

            # Get finger states
            finger_states = detector.get_finger_states(landmarks)

            # Recognize gesture
            gesture = recognizer.recognize(landmarks, finger_states)
            current_gesture = gesture

            # Handle gesture
            action = controller.handle_gesture(gesture)
            if action:
                last_action = action

        # Display info
        cv2.putText(display, "Gesture Control", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if current_gesture:
            cv2.putText(display, f"Gesture: {current_gesture}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if last_action:
            cv2.putText(display, f"Action: {last_action}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Gesture Control", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_demo():
    """Run demo without actual screen control."""
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe required. Install with: pip install mediapipe")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    detector = HandDetector()
    recognizer = GestureRecognizer()

    print("Gesture Recognition Demo (no screen control)")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        hands = detector.detect(frame)

        if hands:
            display = detector.draw_landmarks(display, hands)

            landmarks = hands[0]
            finger_states = detector.get_finger_states(landmarks)
            gesture = recognizer.recognize(landmarks, finger_states)

            # Draw finger states
            fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            for i, (name, state) in enumerate(zip(fingers, finger_states)):
                color = (0, 255, 0) if state else (0, 0, 255)
                cv2.putText(display, f"{name}: {'UP' if state else 'DOWN'}",
                           (10, 100 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if gesture:
                cv2.putText(display, f"Gesture: {gesture}", (10, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(display, "Gesture Demo", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Gesture Demo", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gesture Control System")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--control", action="store_true", help="Enable screen control")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.control:
        run_gesture_control()
    else:
        print("Gesture-Controlled Presentation System")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python main.py --demo     # Demo mode (no screen control)")
        print("  python main.py --control  # Full control mode")
        print()
        print("Requirements:")
        print("  pip install mediapipe pyautogui")
