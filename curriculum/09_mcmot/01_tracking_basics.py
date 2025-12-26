"""
Module 10: MCMOT - Tracking Basics
==================================
Understanding object tracking fundamentals and OpenCV trackers.

Official Docs: https://docs.opencv.org/4.x/d9/df8/group__tracking.html

Topics Covered:
1. Tracking vs Detection
2. OpenCV Built-in Trackers
3. Single Object Tracking Demo
4. Intersection over Union (IoU)
5. Why We Need Re-ID for MOT
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_sample_path

print("=" * 60)
print("Module 10: Tracking Basics")
print("=" * 60)


# =============================================================================
# 1. TRACKING VS DETECTION
# =============================================================================
print("\n--- 1. Tracking vs Detection ---")

concepts = """
Detection vs Tracking:

┌─────────────────────────────────────────────────────────────────────────┐
│  DETECTION                           TRACKING                           │
│  ──────────                          ────────                           │
│  - Process each frame independently  - Uses temporal information        │
│  - Computationally expensive         - Fast (once initialized)          │
│  - Finds ALL objects                 - Follows SPECIFIC objects         │
│  - No identity across frames         - Maintains identity (ID)          │
│  - Handles new appearances           - Can lose track (occlusion)       │
│  - Required: trained model           - Required: initial bbox           │
└─────────────────────────────────────────────────────────────────────────┘

Tracking-by-Detection (Best of Both):
  1. Detect objects periodically (every N frames)
  2. Track between detections (fast)
  3. Re-associate IDs when detecting again

When to Use Which:
  - Detection only: Static scenes, counting objects
  - Tracking only: Known objects, real-time constraints
  - Tracking-by-detection: MOT, MCMOT, surveillance
"""
print(concepts)


# =============================================================================
# 2. OPENCV BUILT-IN TRACKERS
# =============================================================================
print("\n--- 2. OpenCV Built-in Trackers ---")

tracker_info = """
Available Trackers in OpenCV:

┌─────────────────────────────────────────────────────────────────────────┐
│  Tracker    │ Speed      │ Accuracy │ Notes                            │
├─────────────┼────────────┼──────────┼──────────────────────────────────┤
│  MOSSE      │ Very Fast  │ Low      │ Minimum Output Sum of Squared    │
│             │            │          │ Error. Best for high FPS needs   │
├─────────────┼────────────┼──────────┼──────────────────────────────────┤
│  KCF        │ Fast       │ Medium   │ Kernelized Correlation Filter    │
│             │            │          │ Good balance speed/accuracy      │
├─────────────┼────────────┼──────────┼──────────────────────────────────┤
│  CSRT       │ Slower     │ High     │ Discriminative Correlation       │
│             │            │          │ Filter with Channel & Spatial    │
│             │            │          │ Reliability. Best accuracy       │
├─────────────┼────────────┼──────────┼──────────────────────────────────┤
│  MedianFlow │ Fast       │ Medium   │ Tracks forward & backward,       │
│             │            │          │ detects tracking failures        │
└─────────────────────────────────────────────────────────────────────────┘

Creation:
  tracker = cv2.TrackerCSRT_create()  # or TrackerKCF_create(), etc.

Usage:
  tracker.init(frame, bbox)           # Initialize with first frame & bbox
  success, bbox = tracker.update(frame) # Update on new frame
"""
print(tracker_info)


def get_available_trackers():
    """List available trackers in current OpenCV version."""
    trackers = []

    tracker_types = [
        ('CSRT', 'TrackerCSRT_create'),
        ('KCF', 'TrackerKCF_create'),
        ('MOSSE', 'TrackerMOSSE_create'),
        ('MIL', 'TrackerMIL_create'),
    ]

    for name, method in tracker_types:
        if hasattr(cv2, method):
            trackers.append(name)

    return trackers


available = get_available_trackers()
print(f"\nAvailable trackers in OpenCV {cv2.__version__}:")
for t in available:
    print(f"  - {t}")


def create_tracker(tracker_type='CSRT'):
    """Create a tracker by type name."""
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
        return cv2.TrackerMOSSE_create()
    elif tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    else:
        print(f"Unknown tracker type: {tracker_type}, using CSRT")
        return cv2.TrackerCSRT_create()


# =============================================================================
# 3. INTERSECTION OVER UNION (IoU)
# =============================================================================
print("\n--- 3. Intersection over Union (IoU) ---")

iou_info = """
IoU - Key Metric for Object Tracking:

IoU measures overlap between two bounding boxes.

       Box A            Box B           IoU = Area(A ∩ B) / Area(A ∪ B)
    ┌─────────┐      ┌─────────┐
    │         │      │         │       Range: [0, 1]
    │    A    │      │    B    │       0 = No overlap
    │         │      │         │       1 = Perfect overlap
    └─────────┘      └─────────┘

         Overlap
      ┌─────┬───┐
      │     │   │
      │  A  │ B │     IoU = 0.3 (30% overlap)
      │     │   │
      └─────┴───┘

Uses in MOT:
  - Associate detections with tracks
  - Measure tracking accuracy
  - Cost matrix for Hungarian algorithm
"""
print(iou_info)


def compute_iou(box1, box2):
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        box1: (x, y, w, h) format
        box2: (x, y, w, h) format

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2) format
    box1_x1, box1_y1 = x1, y1
    box1_x2, box1_y2 = x1 + w1, y1 + h1

    box2_x1, box2_y1 = x2, y2
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # Intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    # IoU
    if union_area == 0:
        return 0.0

    return inter_area / union_area


# Demo IoU calculation
print("\nIoU Demo:")
box_a = (100, 100, 50, 50)  # (x, y, w, h)
box_b = (120, 110, 50, 50)

iou = compute_iou(box_a, box_b)
print(f"  Box A: {box_a}")
print(f"  Box B: {box_b}")
print(f"  IoU: {iou:.3f}")


def visualize_iou(box1, box2, img_size=(400, 400)):
    """Visualize two boxes and their IoU."""
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Draw boxes
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), -1)
    cv2.rectangle(overlay, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # Draw outlines
    cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
    cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)

    # Add IoU text
    iou = compute_iou(box1, box2)
    cv2.putText(img, f"IoU: {iou:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Box A (Blue)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(img, "Box B (Green)", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img


# =============================================================================
# 4. WHY WE NEED RE-ID FOR MOT
# =============================================================================
print("\n--- 4. Why We Need Re-ID for MOT ---")

reid_motivation = """
Problems with Simple Tracking:

1. OCCLUSION
   ┌────────────────────────────────────────────────────────────────┐
   │  Frame 1      Frame 2       Frame 3       Frame 4             │
   │  ┌───┐        ┌───┐  ┌───┐                ┌───┐               │
   │  │ A │        │ A │  │ B │                │ ? │               │
   │  └───┘        └───┘  └───┘                └───┘               │
   │               Person A hidden behind B     Who is this?       │
   │                                                                │
   │  Without Re-ID: ID is lost after occlusion                    │
   │  With Re-ID: Match appearance → Still ID=A                    │
   └────────────────────────────────────────────────────────────────┘

2. ID SWITCH (Crossing Paths)
   ┌────────────────────────────────────────────────────────────────┐
   │  Frame 1           Frame 2            Frame 3                 │
   │  ┌───┐   ┌───┐    ┌───┐              ┌───┐   ┌───┐           │
   │  │ A │   │ B │    │A/B│ (close)      │ ? │   │ ? │           │
   │  └───┘   └───┘    └───┘              └───┘   └───┘           │
   │                                                                │
   │  Without Re-ID: IoU can match A→B and B→A (ID switch!)       │
   │  With Re-ID: Appearance prevents wrong associations           │
   └────────────────────────────────────────────────────────────────┘

3. RE-ENTRY
   ┌────────────────────────────────────────────────────────────────┐
   │  Camera 1                     Camera 2                        │
   │  ┌───┐                        ┌───┐                           │
   │  │ A │  Person exits ──────>  │ ? │  Same person?             │
   │  └───┘                        └───┘                           │
   │                                                                │
   │  Without Re-ID: New ID assigned                               │
   │  With Re-ID: Match to global gallery → Same ID                │
   └────────────────────────────────────────────────────────────────┘

Re-ID Solution:
  - Extract appearance features from person crops
  - Compare features using cosine distance
  - Match based on appearance similarity
  - Maintain identity through occlusion, across cameras
"""
print(reid_motivation)


# =============================================================================
# 5. SINGLE OBJECT TRACKING DEMO
# =============================================================================
print("\n--- 5. Single Object Tracking Demo ---")


def create_moving_target_video(num_frames=100):
    """Create synthetic video with a moving colored target."""
    frames = []
    h, w = 480, 640

    # Target properties
    target_w, target_h = 60, 80
    x, y = 100, 200
    vx, vy = 5, 2  # Velocity

    for i in range(num_frames):
        # Create frame
        frame = np.ones((h, w, 3), dtype=np.uint8) * 50  # Dark gray background

        # Add some background noise/texture
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Move target (with bouncing)
        x += vx
        y += vy

        if x < 0 or x + target_w > w:
            vx = -vx
            x = max(0, min(x, w - target_w))
        if y < 0 or y + target_h > h:
            vy = -vy
            y = max(0, min(y, h - target_h))

        # Draw target (colored rectangle with pattern)
        cv2.rectangle(frame, (int(x), int(y)),
                     (int(x + target_w), int(y + target_h)),
                     (0, 165, 255), -1)  # Orange fill
        cv2.rectangle(frame, (int(x), int(y)),
                     (int(x + target_w), int(y + target_h)),
                     (0, 100, 200), 2)  # Darker border

        # Add a distinctive pattern
        center_x = int(x + target_w // 2)
        center_y = int(y + target_h // 2)
        cv2.circle(frame, (center_x, center_y), 15, (255, 255, 255), -1)

        frames.append(frame)

    # Initial bounding box
    init_bbox = (100, 200, target_w, target_h)

    return frames, init_bbox


def run_single_tracker_demo(tracker_type='CSRT'):
    """Run single object tracking demo."""
    print(f"\n  Running {tracker_type} tracker demo...")

    # Create synthetic video
    frames, init_bbox = create_moving_target_video(num_frames=100)

    # Create tracker
    tracker = create_tracker(tracker_type)

    # Initialize tracker with first frame
    tracker.init(frames[0], init_bbox)

    # Track through frames
    successes = 0
    total_frames = len(frames)

    result_frames = []

    for i, frame in enumerate(frames):
        # Update tracker
        success, bbox = tracker.update(frame)

        # Draw result
        vis_frame = frame.copy()

        if success:
            successes += 1
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Tracking: {tracker_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "Lost track!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(vis_frame, f"Frame: {i+1}/{total_frames}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        result_frames.append(vis_frame)

    success_rate = successes / total_frames * 100
    print(f"  Success rate: {success_rate:.1f}% ({successes}/{total_frames})")

    return result_frames


# =============================================================================
# 6. TRACKING COMPARISON
# =============================================================================
print("\n--- 6. Tracker Comparison ---")


def compare_trackers():
    """Compare different tracker types on same video."""
    # Create test video
    frames, init_bbox = create_moving_target_video(num_frames=50)

    results = {}

    for tracker_type in ['CSRT', 'KCF', 'MOSSE']:
        if tracker_type not in get_available_trackers():
            continue

        tracker = create_tracker(tracker_type)
        tracker.init(frames[0], init_bbox)

        successes = 0
        ious = []

        for frame in frames[1:]:
            success, bbox = tracker.update(frame)
            if success:
                successes += 1
                # In real scenario, compare with ground truth
                # Here we just count successes

        results[tracker_type] = {
            'success_rate': successes / (len(frames) - 1) * 100,
        }

    print("\nTracker Comparison Results:")
    print("-" * 40)
    for tracker_type, metrics in results.items():
        print(f"  {tracker_type:10s}: {metrics['success_rate']:.1f}% success")


compare_trackers()


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display tracking demonstrations."""
    print("\n" + "=" * 60)
    print("Running Visual Demos...")
    print("=" * 60)

    # 1. IoU visualization
    iou_img = visualize_iou((100, 100, 100, 100), (150, 120, 100, 100))
    cv2.imshow("IoU Visualization", iou_img)

    # 2. Single tracker demo
    result_frames = run_single_tracker_demo('CSRT')

    print("\nPress any key to step through frames, ESC to exit...")

    for frame in result_frames:
        cv2.imshow("Single Object Tracking", frame)
        key = cv2.waitKey(50)
        if key == 27:  # ESC
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running tracking demonstrations...")
    print("=" * 60)
    show_demo()
