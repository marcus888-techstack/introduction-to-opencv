"""
Module 10: MCMOT - Multi-Camera Multi-Object Tracking
=======================================================
Cross-camera tracking using Re-ID for global identity management.

Topics Covered:
1. Multi-Camera Setup
2. Local Per-Camera Tracking
3. Cross-Camera Re-ID Matching
4. Global Track Management
5. Multi-Camera Visualization
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_sample_path

print("=" * 60)
print("Module 10: Multi-Camera Multi-Object Tracking (MCMOT)")
print("=" * 60)


# =============================================================================
# 1. MULTI-CAMERA CONCEPTS
# =============================================================================
print("\n--- 1. Multi-Camera Concepts ---")

mcmot_concepts = """
Multi-Camera Multi-Object Tracking (MCMOT):

Goal: Track persons with CONSISTENT IDs across MULTIPLE cameras.

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│    Camera 1          Camera 2          Camera 3                         │
│       │                 │                 │                             │
│       ▼                 ▼                 ▼                             │
│    ┌──────┐          ┌──────┐          ┌──────┐                        │
│    │ MOT  │          │ MOT  │          │ MOT  │   Local Tracking       │
│    │Local │          │Local │          │Local │   (per camera)         │
│    └──┬───┘          └──┬───┘          └──┬───┘                        │
│       │                 │                 │                             │
│       │ local tracks    │ local tracks    │ local tracks               │
│       │ + features      │ + features      │ + features                 │
│       │                 │                 │                             │
│       └────────────────┬┴─────────────────┘                            │
│                        │                                                │
│                        ▼                                                │
│            ┌───────────────────────────┐                               │
│            │   Global Re-ID Manager    │                               │
│            │                           │                               │
│            │  - Global Gallery         │                               │
│            │  - Cross-camera matching  │                               │
│            │  - Local→Global ID map    │                               │
│            └───────────────────────────┘                               │
│                        │                                                │
│                        ▼                                                │
│              Global Track IDs                                          │
│              (consistent across cameras)                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Challenges:
  - Appearance changes (lighting, angle)
  - Non-overlapping camera views
  - Time delays between cameras
  - Feature drift over time
"""
print(mcmot_concepts)


# =============================================================================
# 2. LOCAL TRACK REPRESENTATION
# =============================================================================
print("\n--- 2. Local Track Representation ---")


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


def cosine_distance(feat1, feat2):
    """Compute cosine distance between two L2-normalized feature vectors."""
    return 1.0 - np.dot(feat1, feat2)


class LocalTrack:
    """Track within a single camera."""

    def __init__(self, track_id, camera_id, bbox, features):
        """
        Initialize local track.

        Args:
            track_id: Local track ID (unique within camera)
            camera_id: ID of the camera
            bbox: (x, y, w, h) bounding box
            features: Re-ID feature vector
        """
        self.local_id = track_id
        self.camera_id = camera_id
        self.bbox = bbox
        self.features = features.copy()
        self.global_id = None  # Assigned by global manager
        self.last_seen = 0
        self.history = []

    def update(self, bbox, features, frame_idx):
        """Update track with new detection."""
        self.bbox = bbox
        # Update features with EMA
        alpha = 0.2
        self.features = (1 - alpha) * self.features + alpha * features
        self.features /= np.linalg.norm(self.features)
        self.last_seen = frame_idx
        self.history.append(bbox)
        if len(self.history) > 30:
            self.history.pop(0)


# =============================================================================
# 3. GLOBAL RE-ID MANAGER
# =============================================================================
print("\n--- 3. Global Re-ID Manager ---")


class GlobalReIDManager:
    """
    Manages global identities across all cameras.
    """

    def __init__(self, distance_threshold=0.5):
        """
        Initialize global manager.

        Args:
            distance_threshold: Max distance for Re-ID match
        """
        self.distance_threshold = distance_threshold
        self.global_gallery = {}  # global_id -> features
        self.next_global_id = 1

        # Mapping: (camera_id, local_id) -> global_id
        self.local_to_global = {}

    def _find_global_match(self, features):
        """
        Find matching global ID for given features.

        Returns:
            (global_id, distance) or (None, None)
        """
        if not self.global_gallery:
            return None, None

        best_id = None
        best_distance = float('inf')

        for global_id, gallery_features in self.global_gallery.items():
            dist = cosine_distance(features, gallery_features)
            if dist < best_distance:
                best_distance = dist
                best_id = global_id

        if best_distance < self.distance_threshold:
            return best_id, best_distance
        else:
            return None, best_distance

    def _create_global_track(self, features):
        """Create new global identity."""
        global_id = self.next_global_id
        self.next_global_id += 1
        self.global_gallery[global_id] = features.copy()
        return global_id

    def _update_global_features(self, global_id, new_features, alpha=0.1):
        """Update global gallery features with EMA."""
        if global_id in self.global_gallery:
            old_features = self.global_gallery[global_id]
            updated = (1 - alpha) * old_features + alpha * new_features
            self.global_gallery[global_id] = updated / np.linalg.norm(updated)

    def assign_global_id(self, camera_id, local_id, features):
        """
        Assign global ID to a local track.

        Args:
            camera_id: Camera identifier
            local_id: Local track ID within camera
            features: Re-ID feature vector

        Returns:
            global_id, is_new
        """
        key = (camera_id, local_id)

        # Check if already assigned
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            self._update_global_features(global_id, features)
            return global_id, False

        # Try to find match in global gallery
        global_id, distance = self._find_global_match(features)

        if global_id is not None:
            # Found match
            self.local_to_global[key] = global_id
            self._update_global_features(global_id, features)
            return global_id, False
        else:
            # Create new global identity
            global_id = self._create_global_track(features)
            self.local_to_global[key] = global_id
            return global_id, True

    def get_global_id(self, camera_id, local_id):
        """Get global ID for a local track."""
        key = (camera_id, local_id)
        return self.local_to_global.get(key, None)

    def get_all_global_ids(self):
        """Get all global IDs in the system."""
        return list(self.global_gallery.keys())


# =============================================================================
# 4. SIMPLE LOCAL TRACKER (PER CAMERA)
# =============================================================================
print("\n--- 4. Per-Camera Local Tracker ---")


class SimpleLocalTracker:
    """
    Simple local tracker for a single camera.
    Uses IoU + Re-ID for matching.
    """

    def __init__(self, camera_id, iou_threshold=0.3, reid_threshold=0.5):
        """
        Initialize local tracker.

        Args:
            camera_id: Unique camera identifier
            iou_threshold: IoU matching threshold
            reid_threshold: Re-ID distance threshold
        """
        self.camera_id = camera_id
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold
        self.tracks = []
        self.next_local_id = 1
        self.frame_count = 0

    def _match_tracks_to_detections(self, detections, det_features):
        """Match existing tracks to new detections."""
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        # Compute cost matrix
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou = compute_iou(track.bbox, det)
                iou_cost = 1 - iou

                reid_cost = cosine_distance(track.features, det_features[j])

                cost[i, j] = 0.5 * iou_cost + 0.5 * reid_cost

        # Greedy matching
        matches = []
        unmatched_tracks = list(range(n_tracks))
        unmatched_dets = list(range(n_dets))

        for i in range(n_tracks):
            if not unmatched_dets:
                break

            best_j = None
            best_cost = float('inf')

            for j in unmatched_dets:
                if cost[i, j] < best_cost:
                    best_cost = cost[i, j]
                    best_j = j

            threshold = max(self.iou_threshold, self.reid_threshold)
            if best_cost < threshold and best_j is not None:
                matches.append((i, best_j))
                unmatched_tracks.remove(i)
                unmatched_dets.remove(best_j)

        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections, det_features):
        """
        Update tracker with new detections.

        Args:
            detections: List of (x, y, w, h) bboxes
            det_features: List of Re-ID feature vectors

        Returns:
            List of active LocalTrack objects
        """
        self.frame_count += 1

        # Match tracks to detections
        matches, unmatched_tracks, unmatched_dets = \
            self._match_tracks_to_detections(detections, det_features)

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                detections[det_idx],
                det_features[det_idx],
                self.frame_count
            )

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = LocalTrack(
                track_id=self.next_local_id,
                camera_id=self.camera_id,
                bbox=detections[det_idx],
                features=det_features[det_idx]
            )
            self.next_local_id += 1
            self.tracks.append(new_track)

        # Remove old tracks
        max_age = 30
        self.tracks = [t for t in self.tracks
                      if self.frame_count - t.last_seen < max_age]

        return self.tracks


# =============================================================================
# 5. MCMOT TRACKER
# =============================================================================
print("\n--- 5. MCMOT Tracker ---")


class MCMOTTracker:
    """
    Multi-Camera Multi-Object Tracker.
    Coordinates multiple cameras with global Re-ID.
    """

    def __init__(self, num_cameras, reid_threshold=0.5):
        """
        Initialize MCMOT tracker.

        Args:
            num_cameras: Number of cameras to track
            reid_threshold: Global Re-ID matching threshold
        """
        self.num_cameras = num_cameras
        self.local_trackers = [
            SimpleLocalTracker(camera_id=i)
            for i in range(num_cameras)
        ]
        self.global_manager = GlobalReIDManager(distance_threshold=reid_threshold)
        self.frame_count = 0

    def update(self, frames, all_detections, all_features):
        """
        Update all cameras with new frames and detections.

        Args:
            frames: List of frames (one per camera)
            all_detections: List of detection lists (per camera)
            all_features: List of feature lists (per camera)

        Returns:
            List of track results per camera
        """
        self.frame_count += 1
        results = []

        # Update each camera's local tracker
        for cam_id in range(self.num_cameras):
            local_tracks = self.local_trackers[cam_id].update(
                all_detections[cam_id],
                all_features[cam_id]
            )

            # Assign global IDs
            for track in local_tracks:
                global_id, is_new = self.global_manager.assign_global_id(
                    cam_id,
                    track.local_id,
                    track.features
                )
                track.global_id = global_id

            results.append(local_tracks)

        return results

    def get_cross_camera_pairs(self):
        """
        Find persons appearing in multiple cameras.

        Returns:
            Dict: global_id -> list of (camera_id, local_id)
        """
        pairs = {}

        for (cam_id, local_id), global_id in self.global_manager.local_to_global.items():
            if global_id not in pairs:
                pairs[global_id] = []
            pairs[global_id].append((cam_id, local_id))

        # Filter to only those appearing in multiple cameras
        multi_cam = {gid: locs for gid, locs in pairs.items()
                    if len(set(c for c, l in locs)) > 1}

        return multi_cam


# =============================================================================
# 6. VISUALIZATION
# =============================================================================
print("\n--- 6. Visualization ---")


def get_color_for_id(track_id, seed=42):
    """Get consistent color for a track ID."""
    np.random.seed(track_id * 123 + seed)
    return tuple(np.random.randint(50, 255, 3).tolist())


def draw_local_tracks(frame, tracks, camera_id):
    """Draw tracks on a single camera frame."""
    result = frame.copy()

    for track in tracks:
        if track.global_id is None:
            color = (128, 128, 128)  # Gray for unassigned
            label = f"L:{track.local_id}"
        else:
            color = get_color_for_id(track.global_id)
            label = f"G:{track.global_id}"

        x, y, w, h = [int(v) for v in track.bbox]

        # Draw box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw label
        cv2.putText(result, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Camera label
    cv2.putText(result, f"Camera {camera_id}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, f"Tracks: {len(tracks)}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def create_multicam_view(frames, all_tracks):
    """
    Create combined multi-camera view.

    Args:
        frames: List of camera frames
        all_tracks: List of track lists per camera

    Returns:
        Combined visualization image
    """
    annotated = []
    for i, (frame, tracks) in enumerate(zip(frames, all_tracks)):
        annotated.append(draw_local_tracks(frame, tracks, i))

    # Resize to same height
    target_height = 300
    resized = []
    for frame in annotated:
        h, w = frame.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(frame, (new_w, target_height)))

    # Combine horizontally
    combined = np.hstack(resized)

    return combined


# =============================================================================
# 7. DEMO
# =============================================================================
print("\n--- 7. Demo ---")


def create_synthetic_multicam_video(num_cameras=2, num_frames=100):
    """
    Create synthetic multi-camera video with persons moving across cameras.
    """
    h, w = 300, 400

    # Define persons (shared across cameras)
    persons = [
        {
            'id': 1,
            'color': (50, 100, 150),
            'cameras': {
                0: {'x': 50, 'y': 80, 'vx': 3, 'vy': 0.5, 'start': 0, 'end': 50},
                1: {'x': 50, 'y': 100, 'vx': 2, 'vy': 0.3, 'start': 30, 'end': 100},
            }
        },
        {
            'id': 2,
            'color': (80, 60, 120),
            'cameras': {
                0: {'x': 300, 'y': 100, 'vx': -2, 'vy': 0.5, 'start': 20, 'end': 80},
                1: {'x': 300, 'y': 80, 'vx': -3, 'vy': 0.2, 'start': 60, 'end': 100},
            }
        },
        {
            'id': 3,
            'color': (100, 150, 50),
            'cameras': {
                0: {'x': 150, 'y': 50, 'vx': 1, 'vy': 1, 'start': 0, 'end': 100},
            }
        },
    ]

    all_frames = [[] for _ in range(num_cameras)]
    all_detections = [[] for _ in range(num_cameras)]
    all_features = [[] for _ in range(num_cameras)]

    for frame_idx in range(num_frames):
        for cam_id in range(num_cameras):
            # Create frame
            frame = np.ones((h, w, 3), dtype=np.uint8) * 180
            frame += np.random.randint(0, 20, frame.shape, dtype=np.uint8)

            frame_dets = []
            frame_feats = []

            for person in persons:
                if cam_id not in person['cameras']:
                    continue

                cam_data = person['cameras'][cam_id]

                if frame_idx < cam_data['start'] or frame_idx > cam_data['end']:
                    continue

                # Update position
                t = frame_idx - cam_data['start']
                x = int(cam_data['x'] + cam_data['vx'] * t)
                y = int(cam_data['y'] + cam_data['vy'] * t)

                pw, ph = 40, 80

                # Boundary check
                x = max(0, min(x, w - pw))
                y = max(0, min(y, h - ph))

                # Draw person
                color = person['color']
                cv2.rectangle(frame, (x, y + 20), (x + pw, y + ph), color, -1)
                cv2.circle(frame, (x + pw // 2, y + 12), 10, (200, 180, 160), -1)

                # Detection
                frame_dets.append((x, y, pw, ph))

                # Generate consistent features for same person
                np.random.seed(person['id'] * 1000 + cam_id)
                base_feat = np.random.randn(512).astype(np.float32)
                base_feat /= np.linalg.norm(base_feat)

                # Add small variation
                noise = np.random.randn(512).astype(np.float32) * 0.1
                feat = base_feat + noise
                feat /= np.linalg.norm(feat)

                frame_feats.append(feat)

            all_frames[cam_id].append(frame)
            all_detections[cam_id].append(frame_dets)
            all_features[cam_id].append(frame_feats)

    return all_frames, all_detections, all_features


def run_mcmot_demo():
    """Run MCMOT demo."""
    print("\n  Creating synthetic multi-camera video...")

    num_cameras = 2
    num_frames = 100

    all_frames, all_detections, all_features = \
        create_synthetic_multicam_video(num_cameras, num_frames)

    print(f"  Created {num_frames} frames for {num_cameras} cameras")

    # Create MCMOT tracker
    tracker = MCMOTTracker(num_cameras=num_cameras, reid_threshold=0.5)

    result_frames = []

    for frame_idx in range(num_frames):
        # Get current frame data for all cameras
        frames = [all_frames[c][frame_idx] for c in range(num_cameras)]
        detections = [all_detections[c][frame_idx] for c in range(num_cameras)]
        features = [all_features[c][frame_idx] for c in range(num_cameras)]

        # Update tracker
        all_tracks = tracker.update(frames, detections, features)

        # Visualize
        combined = create_multicam_view(frames, all_tracks)

        # Add frame info
        cv2.putText(combined, f"Frame: {frame_idx + 1}/{num_frames}", (10, combined.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        result_frames.append(combined)

    # Print cross-camera statistics
    cross_cam = tracker.get_cross_camera_pairs()
    print(f"\n  Persons appearing in multiple cameras: {len(cross_cam)}")
    for gid, locations in cross_cam.items():
        cams = set(c for c, l in locations)
        print(f"    Global ID {gid}: Cameras {cams}")

    return result_frames


def show_demo():
    """Display MCMOT demo."""
    print("\n" + "=" * 60)
    print("Running MCMOT Demo...")
    print("=" * 60)

    result_frames = run_mcmot_demo()

    print("\nPress any key to step, ESC to exit...")

    for frame in result_frames:
        cv2.imshow("MCMOT - Multi-Camera Tracking", frame)
        key = cv2.waitKey(50)
        if key == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running MCMOT demonstrations...")
    print("=" * 60)
    show_demo()
