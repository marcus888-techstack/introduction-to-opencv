"""
Project 4: Real-Time Object Counting & Tracking
================================================
Build a people/vehicle counter for retail analytics, traffic monitoring,
or warehouse management.

Key Concepts:
- Object detection (YOLO)
- Multi-object tracking (MOT)
- Counting line/zone logic
- Direction detection (in/out)
- Analytics dashboard

Official OpenCV References:
- Video Analysis: https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html
- DNN Module: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
"""

import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Ultralytics not installed. Install with: pip install ultralytics")


class SimpleTracker:
    """
    Simple centroid-based object tracker.
    Tracks objects by comparing centroid positions across frames.
    """

    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}  # {id: centroid}
        self.disappeared = {}  # {id: frames_disappeared}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """Register new object with unique ID."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id):
        """Remove object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: List of (x, y, w, h) bounding boxes

        Returns:
            Dict of {object_id: centroid}
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Get centroids from detections
        input_centroids = []
        for (x, y, w, h) in detections:
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))

        # If no existing objects, register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distances
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_c in enumerate(object_centroids):
                for j, inp_c in enumerate(input_centroids):
                    D[i, j] = np.sqrt((obj_c[0] - inp_c[0])**2 +
                                      (obj_c[1] - inp_c[1])**2)

            # Match based on minimum distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > 100:  # Max distance threshold
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched
            unused_rows = set(range(len(object_centroids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols

            # Mark unused existing objects as disappeared
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new objects
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


class CountingLine:
    """
    Counting line to detect objects crossing a boundary.
    """

    def __init__(self, start_point, end_point, direction='horizontal'):
        self.start = start_point
        self.end = end_point
        self.direction = direction

        # Tracking which objects have crossed
        self.previous_positions = {}
        self.count_up = 0
        self.count_down = 0

    def check_crossing(self, object_id, centroid):
        """
        Check if object has crossed the line.

        Returns:
            1 for up/left crossing, -1 for down/right, 0 for no crossing
        """
        if object_id not in self.previous_positions:
            self.previous_positions[object_id] = centroid
            return 0

        prev = self.previous_positions[object_id]
        self.previous_positions[object_id] = centroid

        if self.direction == 'horizontal':
            # Line is horizontal, check Y crossing
            line_y = self.start[1]
            if prev[1] < line_y <= centroid[1]:
                self.count_down += 1
                return -1
            elif prev[1] > line_y >= centroid[1]:
                self.count_up += 1
                return 1
        else:
            # Line is vertical, check X crossing
            line_x = self.start[0]
            if prev[0] < line_x <= centroid[0]:
                self.count_down += 1
                return -1
            elif prev[0] > line_x >= centroid[0]:
                self.count_up += 1
                return 1

        return 0

    def draw(self, image):
        """Draw the counting line on image."""
        cv2.line(image, self.start, self.end, (0, 255, 255), 3)

        # Draw counts
        mid_x = (self.start[0] + self.end[0]) // 2
        mid_y = (self.start[1] + self.end[1]) // 2

        cv2.putText(image, f"In: {self.count_up}", (mid_x - 60, mid_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Out: {self.count_down}", (mid_x - 60, mid_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


class ObjectCounter:
    """
    Main object counting system.
    """

    def __init__(self, target_classes=None):
        """
        Initialize counter.

        Args:
            target_classes: List of class names to count (e.g., ['person', 'car'])
        """
        if YOLO_AVAILABLE:
            print("Loading YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')
            self.use_yolo = True
        else:
            print("Using background subtraction (YOLO not available)")
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.use_yolo = False

        self.tracker = SimpleTracker()
        self.counting_line = None
        self.target_classes = target_classes or ['person']

        # YOLO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def set_counting_line(self, start, end, direction='horizontal'):
        """Set the counting line position."""
        self.counting_line = CountingLine(start, end, direction)

    def detect_objects(self, frame):
        """Detect objects in frame."""
        if self.use_yolo:
            return self._detect_yolo(frame)
        else:
            return self._detect_background_sub(frame)

    def _detect_yolo(self, frame):
        """Detect using YOLO."""
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = self.class_names[cls]

                if class_name in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    if conf > 0.5:
                        detections.append((x1, y1, x2 - x1, y2 - y1))

        return detections

    def _detect_background_sub(self, frame):
        """Detect using background subtraction (fallback)."""
        fg_mask = self.bg_subtractor.apply(frame)

        # Threshold and find contours
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Min area threshold
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x, y, w, h))

        return detections

    def process_frame(self, frame):
        """
        Process a single frame.

        Returns:
            Annotated frame with detections and counts
        """
        display = frame.copy()

        # Detect objects
        detections = self.detect_objects(frame)

        # Update tracker
        tracked_objects = self.tracker.update(detections)

        # Draw detections and check crossings
        for object_id, centroid in tracked_objects.items():
            # Draw centroid
            cv2.circle(display, centroid, 5, (0, 255, 0), -1)
            cv2.putText(display, f"ID {object_id}", (centroid[0] - 20, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check line crossing
            if self.counting_line:
                self.counting_line.check_crossing(object_id, centroid)

        # Draw bounding boxes
        for (x, y, w, h) in detections:
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw counting line
        if self.counting_line:
            self.counting_line.draw(display)

        # Show current count
        cv2.putText(display, f"Currently Tracking: {len(tracked_objects)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return display

    def get_counts(self):
        """Get current counts."""
        if self.counting_line:
            return {
                'in': self.counting_line.count_up,
                'out': self.counting_line.count_down,
                'total': self.counting_line.count_up + self.counting_line.count_down
            }
        return {'tracking': len(self.tracker.objects)}


def run_video(video_source=0, target_classes=None):
    """
    Run object counting on video.

    Args:
        video_source: Camera ID or video file path
        target_classes: Classes to count
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Could not open video: {video_source}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize counter
    counter = ObjectCounter(target_classes)

    # Set counting line in middle of frame
    counter.set_counting_line(
        start=(0, height // 2),
        end=(width, height // 2),
        direction='horizontal'
    )

    print("Object Counter Started")
    print(f"Counting: {target_classes or ['all objects']}")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        display = counter.process_frame(frame)

        cv2.imshow("Object Counter", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print final counts
    counts = counter.get_counts()
    print("\n=== Final Counts ===")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    cap.release()
    cv2.destroyAllWindows()


def create_demo():
    """Create demo with moving objects."""
    print("Demo: Simulating moving objects...")

    # Create blank frame
    width, height = 800, 600
    counter = ObjectCounter()
    counter.set_counting_line(
        start=(0, height // 2),
        end=(width, height // 2),
        direction='horizontal'
    )

    # Simulate objects
    objects = [
        {'pos': [100, 100], 'vel': [5, 3], 'size': 40},
        {'pos': [300, 50], 'vel': [3, 4], 'size': 35},
        {'pos': [500, 150], 'vel': [-4, 5], 'size': 45},
    ]

    print("Press Q to quit demo")

    for _ in range(500):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50

        # Update and draw objects
        detections = []
        for obj in objects:
            # Update position
            obj['pos'][0] += obj['vel'][0]
            obj['pos'][1] += obj['vel'][1]

            # Bounce off walls
            if obj['pos'][0] <= 0 or obj['pos'][0] >= width - obj['size']:
                obj['vel'][0] *= -1
            if obj['pos'][1] <= 0 or obj['pos'][1] >= height - obj['size']:
                obj['vel'][1] *= -1

            # Draw and add to detections
            x, y = int(obj['pos'][0]), int(obj['pos'][1])
            s = obj['size']
            cv2.rectangle(frame, (x, y), (x + s, y + s), (100, 100, 200), -1)
            detections.append((x, y, s, s))

        # Update tracker
        tracked = counter.tracker.update(detections)

        # Check crossings and draw
        for obj_id, centroid in tracked.items():
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            cv2.putText(frame, str(obj_id), (centroid[0] - 5, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if counter.counting_line:
                counter.counting_line.check_crossing(obj_id, centroid)

        # Draw counting line
        if counter.counting_line:
            counter.counting_line.draw(frame)

        cv2.putText(frame, f"Tracking: {len(tracked)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Demo", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    counts = counter.get_counts()
    print("\n=== Demo Counts ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Object Counting System")
    parser.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--camera", action="store_true", help="Use webcam")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--classes", nargs='+', default=['person'],
                       help="Classes to count (default: person)")

    args = parser.parse_args()

    if args.demo:
        create_demo()
    elif args.video:
        run_video(args.video, args.classes)
    elif args.camera:
        run_video(0, args.classes)
    else:
        print("Object Counting System")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python main.py --demo                    # Run demo")
        print("  python main.py --camera                  # Use webcam")
        print("  python main.py --video FILE              # Process video")
        print("  python main.py --camera --classes person car  # Count specific classes")
