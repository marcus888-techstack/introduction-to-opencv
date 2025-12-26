"""
Module 10: MCMOT - Multi-Object Tracking with Re-ID
=====================================================
MOT combining Kalman Filter motion prediction with Re-ID appearance matching.

Official Docs:
- Kalman Filter: https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html

Topics Covered:
1. Kalman Filter for Motion Prediction
2. Hungarian Algorithm for Assignment
3. Combined Cost Matrix (IoU + Re-ID)
4. Track Lifecycle Management
5. DeepSORT-style Tracking
"""

import cv2
import numpy as np
import os
import sys

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Using greedy matching instead.")
    print("Install with: pip install scipy")

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_sample_path

# Import from previous modules in same directory
# Note: When running standalone, these may need adjustment
try:
    from tracking_basics_01 import compute_iou
except ImportError:
    # Define locally if import fails
    def compute_iou(box1, box2):
        """Compute IoU between two boxes in (x, y, w, h) format."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

try:
    from person_reid_03 import extract_features, cosine_distance, PersonReID
except ImportError:
    # Define locally if import fails
    def extract_features(person_crop, reid_net):
        """Extract Re-ID features (fallback: random features)."""
        if reid_net is None:
            features = np.random.randn(512).astype(np.float32)
            return features / np.linalg.norm(features)
        # Full implementation in 03_person_reid.py
        return np.random.randn(512).astype(np.float32)

    def cosine_distance(feat1, feat2):
        """Compute cosine distance."""
        return 1.0 - np.dot(feat1, feat2)

print("=" * 60)
print("Module 10: Multi-Object Tracking with Re-ID")
print("=" * 60)


# =============================================================================
# 1. KALMAN FILTER FOR MOTION PREDICTION
# =============================================================================
print("\n--- 1. Kalman Filter for Motion Prediction ---")

kalman_info = """
Kalman Filter - Predicting Object Motion:

State Vector (8D):
  [x, y, w, h, vx, vy, vw, vh]
   │  │  │  │   │   │   │   └── Height velocity
   │  │  │  │   │   │   └────── Width velocity
   │  │  │  │   │   └────────── Y velocity
   │  │  │  │   └────────────── X velocity
   │  │  │  └────────────────── Height
   │  │  └───────────────────── Width
   │  └──────────────────────── Y position (center)
   └─────────────────────────── X position (center)

Measurement Vector (4D):
  [x, y, w, h]  (from detection)

Prediction Step:
  x_pred = A × x_prev   (predict next state)
  P_pred = A × P × A^T + Q  (predict covariance)

Update/Correction Step:
  K = P × H^T × (H × P × H^T + R)^-1  (Kalman gain)
  x = x_pred + K × (z - H × x_pred)    (update state)
  P = (I - K × H) × P                  (update covariance)

OpenCV Usage:
  kf = cv2.KalmanFilter(8, 4)   # 8 state dims, 4 measurement dims
  kf.predict()                   # Predict next state
  kf.correct(measurement)        # Update with detection
"""
print(kalman_info)


def create_kalman_filter():
    """
    Create a Kalman Filter for bounding box tracking.

    State: [x, y, w, h, vx, vy, vw, vh]
    Measurement: [x, y, w, h]

    Returns:
        Configured KalmanFilter
    """
    kf = cv2.KalmanFilter(8, 4)  # 8 state dims, 4 measurement dims

    # Transition matrix A (state evolution)
    # x_new = x + vx*dt, etc.
    dt = 1.0  # time step
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, dt, 0, 0, 0],
        [0, 1, 0, 0, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0, dt, 0],
        [0, 0, 0, 1, 0, 0, 0, dt],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)

    # Measurement matrix H (we observe x, y, w, h)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Process noise covariance Q
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03

    # Measurement noise covariance R
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

    # Initial state covariance
    kf.errorCovPost = np.eye(8, dtype=np.float32) * 10

    return kf


def init_kalman_state(kf, bbox):
    """
    Initialize Kalman filter state from bounding box.

    Args:
        kf: KalmanFilter object
        bbox: (x, y, w, h) bounding box

    Returns:
        Initialized KalmanFilter
    """
    x, y, w, h = bbox
    cx = x + w / 2  # center x
    cy = y + h / 2  # center y

    kf.statePost = np.array([
        [cx], [cy], [w], [h],
        [0], [0], [0], [0]  # Initial velocities = 0
    ], dtype=np.float32)

    return kf


def predict_bbox(kf):
    """
    Predict next bounding box.

    Args:
        kf: KalmanFilter object

    Returns:
        Predicted (x, y, w, h) or None if invalid
    """
    prediction = kf.predict()

    cx = prediction[0, 0]
    cy = prediction[1, 0]
    w = prediction[2, 0]
    h = prediction[3, 0]

    # Convert center to top-left
    x = cx - w / 2
    y = cy - h / 2

    # Validate
    if w <= 0 or h <= 0:
        return None

    return (int(x), int(y), int(w), int(h))


def update_kalman(kf, bbox):
    """
    Update Kalman filter with detection.

    Args:
        kf: KalmanFilter object
        bbox: (x, y, w, h) detection

    Returns:
        Corrected (x, y, w, h)
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2

    measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
    corrected = kf.correct(measurement)

    cx = corrected[0, 0]
    cy = corrected[1, 0]
    w = corrected[2, 0]
    h = corrected[3, 0]

    x = cx - w / 2
    y = cy - h / 2

    return (int(x), int(y), int(w), int(h))


# =============================================================================
# 2. HUNGARIAN ALGORITHM FOR ASSIGNMENT
# =============================================================================
print("\n--- 2. Hungarian Algorithm for Assignment ---")

hungarian_info = """
Hungarian Algorithm - Optimal Assignment:

Problem: Match N tracks to M detections optimally

Cost Matrix:
              Det 0   Det 1   Det 2
    Track 0 [  0.2     0.8     0.9  ]
    Track 1 [  0.7     0.1     0.6  ]
    Track 2 [  0.9     0.5     0.3  ]

Solution: Minimize total cost
    Track 0 → Det 0 (cost 0.2)
    Track 1 → Det 1 (cost 0.1)
    Track 2 → Det 2 (cost 0.3)
    Total: 0.6

scipy.optimize.linear_sum_assignment(cost_matrix)
    Returns: (row_indices, col_indices)

Unmatched handling:
    - Unmatched tracks: Lost or occluded → increment age
    - Unmatched detections: New objects → create tracks
"""
print(hungarian_info)


def hungarian_matching(cost_matrix, threshold=0.7):
    """
    Solve optimal assignment using Hungarian algorithm.

    Args:
        cost_matrix: NxM cost matrix (tracks x detections)
        threshold: Maximum cost to consider a valid match

    Returns:
        matches: List of (track_idx, det_idx) pairs
        unmatched_tracks: List of track indices
        unmatched_dets: List of detection indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    if SCIPY_AVAILABLE:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    else:
        # Greedy fallback
        row_indices = []
        col_indices = []
        used_cols = set()
        for i in range(cost_matrix.shape[0]):
            best_j = -1
            best_cost = float('inf')
            for j in range(cost_matrix.shape[1]):
                if j not in used_cols and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                row_indices.append(i)
                col_indices.append(best_j)
                used_cols.add(best_j)
        row_indices = np.array(row_indices)
        col_indices = np.array(col_indices)

    # Filter by threshold
    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_dets = list(range(cost_matrix.shape[1]))

    for i, j in zip(row_indices, col_indices):
        if cost_matrix[i, j] < threshold:
            matches.append((i, j))
            if i in unmatched_tracks:
                unmatched_tracks.remove(i)
            if j in unmatched_dets:
                unmatched_dets.remove(j)

    return matches, unmatched_tracks, unmatched_dets


# =============================================================================
# 3. TRACK CLASS
# =============================================================================
print("\n--- 3. Track Lifecycle ---")


class Track:
    """
    Single object track with motion and appearance.
    """

    _next_id = 1

    def __init__(self, bbox, features=None):
        """
        Initialize a new track.

        Args:
            bbox: Initial (x, y, w, h) bounding box
            features: Initial Re-ID features (optional)
        """
        self.id = Track._next_id
        Track._next_id += 1

        # Kalman filter for motion
        self.kf = create_kalman_filter()
        init_kalman_state(self.kf, bbox)

        self.bbox = bbox

        # Appearance features (for Re-ID)
        self.features = features
        self.features_history = []  # Keep last N features
        self.max_features = 10

        # Track state
        self.hits = 1  # Detection matches
        self.age = 0  # Frames since creation
        self.time_since_update = 0  # Frames since last detection

        # Track history for visualization
        self.history = [self._get_center()]

    def _get_center(self):
        """Get center point of current bbox."""
        x, y, w, h = self.bbox
        return (int(x + w / 2), int(y + h / 2))

    def predict(self):
        """Predict next position."""
        self.age += 1
        self.time_since_update += 1

        predicted = predict_bbox(self.kf)
        if predicted is not None:
            self.bbox = predicted

        return self.bbox

    def update(self, bbox, features=None):
        """
        Update track with new detection.

        Args:
            bbox: Matched detection bbox
            features: Updated Re-ID features
        """
        self.hits += 1
        self.time_since_update = 0

        # Update Kalman filter
        self.bbox = update_kalman(self.kf, bbox)

        # Update appearance features (exponential moving average)
        if features is not None:
            if self.features is None:
                self.features = features.copy()
            else:
                alpha = 0.2  # Weight for new features
                self.features = (1 - alpha) * self.features + alpha * features
                self.features /= np.linalg.norm(self.features)

            # Keep feature history
            self.features_history.append(features.copy())
            if len(self.features_history) > self.max_features:
                self.features_history.pop(0)

        # Update history
        self.history.append(self._get_center())
        if len(self.history) > 50:
            self.history.pop(0)

    def is_confirmed(self, n_init=3):
        """Check if track is confirmed (enough hits)."""
        return self.hits >= n_init

    def is_deleted(self, max_age=30):
        """Check if track should be deleted (too long without update)."""
        return self.time_since_update > max_age


# =============================================================================
# 4. MOT TRACKER
# =============================================================================
print("\n--- 4. MOT Tracker (DeepSORT-style) ---")


class MOTTracker:
    """
    Multi-Object Tracker with Re-ID appearance matching.
    """

    def __init__(self, reid_net=None,
                 iou_threshold=0.3,
                 reid_threshold=0.5,
                 max_age=30,
                 n_init=3):
        """
        Initialize MOT tracker.

        Args:
            reid_net: OpenCV DNN Re-ID network
            iou_threshold: IoU matching threshold
            reid_threshold: Re-ID distance threshold
            max_age: Frames before deleting lost track
            n_init: Hits needed to confirm track
        """
        self.reid_net = reid_net
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold
        self.max_age = max_age
        self.n_init = n_init

        self.tracks = []
        self.frame_count = 0

        # Cost weights
        self.lambda_iou = 0.5
        self.lambda_reid = 0.5

    def _extract_features(self, image, bbox):
        """Extract Re-ID features from detection."""
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if self.reid_net is not None:
            return extract_features(crop, self.reid_net)
        else:
            # Random features for demo
            feat = np.random.randn(512).astype(np.float32)
            return feat / np.linalg.norm(feat)

    def _compute_cost_matrix(self, detections, det_features):
        """
        Compute combined IoU + Re-ID cost matrix.

        Args:
            detections: List of (x, y, w, h) bboxes
            det_features: List of feature vectors

        Returns:
            Cost matrix (n_tracks x n_detections)
        """
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            return np.empty((n_tracks, n_dets))

        cost = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                # IoU cost (1 - IoU)
                iou = compute_iou(track.bbox, det)
                iou_cost = 1 - iou

                # Re-ID cost (cosine distance)
                if track.features is not None and det_features[j] is not None:
                    reid_cost = cosine_distance(track.features, det_features[j])
                else:
                    reid_cost = 0.5  # Neutral if no features

                # Combined cost
                cost[i, j] = self.lambda_iou * iou_cost + self.lambda_reid * reid_cost

        return cost

    def update(self, image, detections):
        """
        Update tracker with new frame and detections.

        Args:
            image: Current frame (BGR)
            detections: List of (x, y, w, h) person detections

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        # Step 1: Predict new positions for all tracks
        for track in self.tracks:
            track.predict()

        # Step 2: Extract features for detections
        det_features = []
        for det in detections:
            feat = self._extract_features(image, det)
            det_features.append(feat)

        # Step 3: Compute cost matrix and solve assignment
        cost_matrix = self._compute_cost_matrix(detections, det_features)

        matches, unmatched_tracks, unmatched_dets = hungarian_matching(
            cost_matrix,
            threshold=max(self.iou_threshold, self.reid_threshold)
        )

        # Step 4: Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                detections[det_idx],
                det_features[det_idx]
            )

        # Step 5: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                detections[det_idx],
                det_features[det_idx]
            )
            self.tracks.append(new_track)

        # Step 6: Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]

        # Return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed(self.n_init)]


# =============================================================================
# 5. VISUALIZATION
# =============================================================================
print("\n--- 5. Visualization ---")


def draw_tracks(image, tracks, draw_history=True):
    """
    Draw tracks on image.

    Args:
        image: Input image
        tracks: List of Track objects
        draw_history: Whether to draw track trails

    Returns:
        Annotated image
    """
    result = image.copy()

    # Color palette for tracks
    np.random.seed(42)
    colors = {}

    for track in tracks:
        if track.id not in colors:
            colors[track.id] = tuple(np.random.randint(50, 255, 3).tolist())

        color = colors[track.id]
        x, y, w, h = track.bbox

        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw ID
        label = f"ID: {track.id}"
        cv2.putText(result, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw track history (trail)
        if draw_history and len(track.history) > 1:
            for i in range(1, len(track.history)):
                pt1 = track.history[i - 1]
                pt2 = track.history[i]
                # Fade older points
                alpha = i / len(track.history)
                thickness = int(1 + 2 * alpha)
                cv2.line(result, pt1, pt2, color, thickness)

    return result


# =============================================================================
# 6. DEMO
# =============================================================================
print("\n--- 6. Demo ---")


def create_synthetic_video_with_persons(num_frames=100):
    """Create synthetic video with moving person-like objects."""
    frames = []
    h, w = 480, 640

    # Define persons with initial positions and velocities
    persons = [
        {'x': 50, 'y': 100, 'w': 60, 'h': 120, 'vx': 3, 'vy': 1, 'color': (50, 100, 150)},
        {'x': 500, 'y': 200, 'w': 55, 'h': 110, 'vx': -2, 'vy': 1, 'color': (80, 60, 120)},
        {'x': 300, 'y': 50, 'w': 50, 'h': 100, 'vx': 1, 'vy': 2, 'color': (100, 150, 50)},
    ]

    detections_per_frame = []

    for frame_idx in range(num_frames):
        # Create frame
        frame = np.ones((h, w, 3), dtype=np.uint8) * 200

        # Add some texture
        noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        frame_detections = []

        for person in persons:
            # Update position
            person['x'] += person['vx']
            person['y'] += person['vy']

            # Bounce off walls
            if person['x'] < 0 or person['x'] + person['w'] > w:
                person['vx'] = -person['vx']
            if person['y'] < 0 or person['y'] + person['h'] > h:
                person['vy'] = -person['vy']

            person['x'] = max(0, min(person['x'], w - person['w']))
            person['y'] = max(0, min(person['y'], h - person['h']))

            # Draw person
            x, y = int(person['x']), int(person['y'])
            pw, ph = person['w'], person['h']
            color = person['color']

            # Body
            cv2.rectangle(frame, (x, y + 30), (x + pw, y + ph), color, -1)

            # Head
            head_cx = x + pw // 2
            head_cy = y + 20
            cv2.circle(frame, (head_cx, head_cy), 18, (200, 180, 160), -1)

            # Add detection (with some noise)
            noise_x = np.random.randint(-3, 4)
            noise_y = np.random.randint(-3, 4)
            frame_detections.append((x + noise_x, y + noise_y, pw, ph))

        frames.append(frame)
        detections_per_frame.append(frame_detections)

    return frames, detections_per_frame


def run_mot_demo():
    """Run MOT tracking demo."""
    print("\n  Running MOT demo with synthetic video...")

    # Create tracker
    tracker = MOTTracker(
        reid_net=None,  # Use random features for demo
        iou_threshold=0.3,
        reid_threshold=0.5,
        max_age=15,
        n_init=2
    )

    # Create synthetic video
    frames, detections_per_frame = create_synthetic_video_with_persons(num_frames=100)

    print(f"  Created {len(frames)} frames with {len(detections_per_frame[0])} persons")

    result_frames = []

    for i, (frame, detections) in enumerate(zip(frames, detections_per_frame)):
        # Update tracker
        active_tracks = tracker.update(frame, detections)

        # Draw results
        vis_frame = draw_tracks(frame, active_tracks, draw_history=True)

        # Add info
        cv2.putText(vis_frame, f"Frame: {i+1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_frame, f"Tracks: {len(active_tracks)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        result_frames.append(vis_frame)

    return result_frames


def show_demo():
    """Display MOT demo."""
    print("\n" + "=" * 60)
    print("Running MOT Demo...")
    print("=" * 60)

    result_frames = run_mot_demo()

    print("\nPress any key to step, ESC to exit...")

    for frame in result_frames:
        cv2.imshow("MOT Tracking", frame)
        key = cv2.waitKey(50)
        if key == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running MOT demonstrations...")
    print("=" * 60)
    show_demo()
