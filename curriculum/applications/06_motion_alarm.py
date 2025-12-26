"""
Application 06: Motion Detection Alarm
======================================
Security camera-style motion detection with alerts.

Techniques Used:
- Background subtraction (MOG2, KNN)
- Frame differencing
- Contour detection
- Bounding boxes

Official Docs:
- https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
"""

import cv2
import numpy as np
from datetime import datetime
import time
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_video


class MotionDetector:
    """
    Motion detection using various methods.
    """

    def __init__(self, method='mog2', sensitivity=500):
        self.method = method
        self.sensitivity = sensitivity  # Minimum contour area
        self.motion_detected = False
        self.motion_regions = []

        # Initialize background subtractor
        if method == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )
        elif method == 'knn':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=500, dist2Threshold=400, detectShadows=True
            )
        else:  # frame_diff
            self.prev_frame = None

    def detect(self, frame):
        """
        Detect motion in frame.
        Returns: (motion_detected, motion_regions, mask)
        """
        self.motion_detected = False
        self.motion_regions = []

        if self.method in ['mog2', 'knn']:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)

            # Remove shadows (gray = 127)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        else:  # frame_diff
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.prev_frame is None:
                self.prev_frame = gray
                return False, [], np.zeros_like(gray)

            # Frame difference
            diff = cv2.absdiff(self.prev_frame, gray)
            _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            self.prev_frame = gray

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        for contour in contours:
            if cv2.contourArea(contour) >= self.sensitivity:
                x, y, w, h = cv2.boundingRect(contour)
                self.motion_regions.append((x, y, w, h))
                self.motion_detected = True

        return self.motion_detected, self.motion_regions, fg_mask

    def draw_regions(self, frame, color=(0, 255, 0), thickness=2):
        """
        Draw motion regions on frame.
        """
        result = frame.copy()

        for (x, y, w, h) in self.motion_regions:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        return result


class MotionAlarm:
    """
    Motion alarm system with recording and notifications.
    """

    def __init__(self, cooldown=5.0):
        self.detector = MotionDetector(method='mog2')
        self.cooldown = cooldown  # Seconds between alerts
        self.last_alert_time = 0
        self.recording = False
        self.video_writer = None
        self.alert_callback = None

    def set_alert_callback(self, callback):
        """Set function to call when motion detected."""
        self.alert_callback = callback

    def should_alert(self):
        """Check if enough time passed since last alert."""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.cooldown:
            self.last_alert_time = current_time
            return True
        return False

    def process_frame(self, frame):
        """
        Process frame and handle alerts.
        """
        motion, regions, mask = self.detector.detect(frame)

        result = self.detector.draw_regions(frame)

        # Status overlay
        status_color = (0, 0, 255) if motion else (0, 255, 0)
        status_text = "MOTION DETECTED!" if motion else "Monitoring..."

        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, timestamp, (10, result.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Trigger alert
        if motion and self.should_alert():
            if self.alert_callback:
                self.alert_callback(frame, regions)
            print(f"[ALERT] Motion detected at {timestamp}")

        return result, mask, motion

    def start_recording(self, filename, frame_size, fps=20.0):
        """Start recording video."""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.recording = True
        print(f"Recording started: {filename}")

    def stop_recording(self):
        """Stop recording video."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("Recording stopped")

    def record_frame(self, frame):
        """Record frame if recording is active."""
        if self.recording and self.video_writer:
            self.video_writer.write(frame)


def save_snapshot(frame, regions):
    """
    Callback to save snapshot when motion detected.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"motion_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"  Snapshot saved: {filename}")


def interactive_alarm():
    """
    Interactive motion alarm with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== Motion Detection Alarm ===")
    print("Controls:")
    print("  '1' - MOG2 background subtraction")
    print("  '2' - KNN background subtraction")
    print("  '3' - Frame differencing")
    print("  '+' - Increase sensitivity (less sensitive)")
    print("  '-' - Decrease sensitivity (more sensitive)")
    print("  'r' - Toggle recording")
    print("  's' - Enable snapshot on motion")
    print("  'q' - Quit")
    print("==============================\n")

    alarm = MotionAlarm(cooldown=3.0)
    method = 'mog2'
    snapshot_enabled = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, mask, motion = alarm.process_frame(frame)

        # Display sensitivity
        cv2.putText(result, f"Sensitivity: {alarm.detector.sensitivity}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, f"Method: {method}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if alarm.recording:
            cv2.circle(result, (result.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(result, "REC", (result.shape[1] - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            alarm.record_frame(result)

        cv2.imshow("Motion Alarm", result)
        cv2.imshow("Motion Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            method = 'mog2'
            alarm.detector = MotionDetector(method='mog2',
                                           sensitivity=alarm.detector.sensitivity)
        elif key == ord('2'):
            method = 'knn'
            alarm.detector = MotionDetector(method='knn',
                                           sensitivity=alarm.detector.sensitivity)
        elif key == ord('3'):
            method = 'frame_diff'
            alarm.detector = MotionDetector(method='frame_diff',
                                           sensitivity=alarm.detector.sensitivity)
        elif key == ord('+') or key == ord('='):
            alarm.detector.sensitivity += 200
        elif key == ord('-'):
            alarm.detector.sensitivity = max(100, alarm.detector.sensitivity - 200)
        elif key == ord('r'):
            if alarm.recording:
                alarm.stop_recording()
            else:
                h, w = frame.shape[:2]
                alarm.start_recording("motion_recording.avi", (w, h))
        elif key == ord('s'):
            snapshot_enabled = not snapshot_enabled
            if snapshot_enabled:
                alarm.set_alert_callback(save_snapshot)
                print("Snapshot on motion: ENABLED")
            else:
                alarm.set_alert_callback(None)
                print("Snapshot on motion: DISABLED")

    alarm.stop_recording()
    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with real video or synthetic motion.
    """
    print("\n=== Motion Detection Demo ===\n")

    # Try to load real video
    video_path = get_video("vtest.avi")
    cap = None

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"Using sample video: vtest.avi")
        else:
            cap = None

    # Also try slow_traffic video
    if cap is None:
        video_path = get_video("slow_traffic_small.mp4")
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                print(f"Using sample video: slow_traffic_small.mp4")
            else:
                cap = None

    if cap is not None:
        # Use real video
        detector = MotionDetector(method='mog2', sensitivity=500)
        print("Press 'q' to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue

            motion, regions, mask = detector.detect(frame)
            result = detector.draw_regions(frame)

            status = "MOTION DETECTED!" if motion else "No motion"
            color = (0, 0, 255) if motion else (0, 255, 0)
            cv2.putText(result, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Motion Demo", result)
            cv2.imshow("Mask", mask)

            if cv2.waitKey(30) == ord('q'):
                break

        cap.release()
    else:
        # Fallback: synthetic motion
        print("No sample video found. Using synthetic animation.")
        print("Run: python curriculum/sample_data/download_samples.py")

        bg = np.ones((400, 600, 3), dtype=np.uint8) * 200
        cv2.rectangle(bg, (50, 50), (150, 150), (100, 100, 100), -1)

        detector = MotionDetector(method='frame_diff', sensitivity=500)
        obj_x, direction = 200, 1

        for _ in range(200):
            frame = bg.copy()
            obj_x += direction * 5
            if obj_x > 500 or obj_x < 100:
                direction *= -1

            cv2.circle(frame, (obj_x, 200), 30, (0, 0, 255), -1)
            motion, regions, mask = detector.detect(frame)
            result = detector.draw_regions(frame)

            status = "MOTION!" if motion else "No motion"
            cv2.putText(result, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 0, 255) if motion else (0, 255, 0), 2)

            cv2.imshow("Motion Demo", result)
            cv2.imshow("Mask", mask)

            if cv2.waitKey(50) == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 06: Motion Detection Alarm")
    print("=" * 60)

    try:
        interactive_alarm()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
