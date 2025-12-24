"""
Extra Module: Object Tracking (opencv-contrib)
===============================================
Tracking objects across video frames.

Note: Some trackers require opencv-contrib-python

Topics Covered:
1. Single Object Trackers
2. Multi-Object Tracking
3. Tracker Comparison
4. Deep Learning Trackers
"""

import cv2
import numpy as np

print("=" * 60)
print("Extra Module: Object Tracking")
print("=" * 60)


# =============================================================================
# 1. AVAILABLE TRACKERS
# =============================================================================
print("\n--- 1. Available Trackers ---")

trackers_info = """
OpenCV Object Trackers:

Legacy Trackers (cv2.legacy module):
  - BOOSTING     - Based on AdaBoost
  - MIL          - Multiple Instance Learning
  - KCF          - Kernelized Correlation Filters
  - TLD          - Tracking, Learning, Detection
  - MEDIANFLOW   - Based on optical flow
  - MOSSE        - Very fast, basic accuracy
  - CSRT         - Discriminative Correlation Filter

Modern Trackers (cv2 module):
  - TrackerMIL_create()
  - TrackerGOTURN_create()  - Deep learning based
  - TrackerDaSiamRPN_create() - Deep learning based
  - TrackerNano_create()    - Lightweight DNN tracker

Creating Trackers:
  tracker = cv2.TrackerKCF_create()  # or other tracker
  tracker.init(frame, bbox)          # Initialize with bounding box
  success, bbox = tracker.update(frame)  # Update in new frame
"""
print(trackers_info)


# Check available trackers
available_trackers = []
tracker_creators = [
    ('KCF', lambda: cv2.TrackerKCF_create()),
    ('CSRT', lambda: cv2.TrackerCSRT_create()),
    ('MIL', lambda: cv2.TrackerMIL_create()),
]

for name, creator in tracker_creators:
    try:
        t = creator()
        available_trackers.append(name)
    except AttributeError:
        pass

print(f"\nAvailable trackers: {available_trackers}")


# =============================================================================
# 2. CREATE TEST VIDEO
# =============================================================================
print("\n--- 2. Creating Test Data ---")


def create_tracking_frames(num_frames=60):
    """Create video frames with moving object."""
    frames = []
    h, w = 300, 400

    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (80, 80, 80)

        # Moving object (sinusoidal path)
        x = int(100 + 200 * i / num_frames)
        y = int(150 + 50 * np.sin(i * 0.2))

        # Draw object
        cv2.rectangle(frame, (x-25, y-25), (x+25, y+25), (0, 255, 0), -1)
        cv2.putText(frame, "TARGET", (x-20, y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Add some distractors
        cv2.circle(frame, (50, 100), 20, (255, 0, 0), -1)
        cv2.circle(frame, (350, 200), 20, (0, 0, 255), -1)

        frames.append(frame)

    return frames


frames = create_tracking_frames()
print(f"Created {len(frames)} test frames")


# =============================================================================
# 3. SINGLE OBJECT TRACKING
# =============================================================================
print("\n--- 3. Single Object Tracking ---")


def track_object(frames, tracker_type='KCF'):
    """Track object through frames."""
    # Select tracker
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    else:
        print(f"Unknown tracker: {tracker_type}")
        return []

    # Initial bounding box (x, y, w, h)
    # Object starts around (75, 125)
    bbox = (75, 125, 50, 50)

    # Initialize tracker
    tracker.init(frames[0], bbox)

    tracked_frames = []
    bboxes = [bbox]

    for frame in frames[1:]:
        # Update tracker
        success, bbox = tracker.update(frame)

        # Draw result
        display = frame.copy()
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(display, f"{tracker_type}: Tracking", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            bboxes.append(bbox)
        else:
            cv2.putText(display, f"{tracker_type}: Lost", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            bboxes.append(None)

        tracked_frames.append(display)

    return tracked_frames, bboxes


# Run tracking
if 'KCF' in available_trackers:
    tracked_frames, bboxes = track_object(frames, 'KCF')
    success_count = sum(1 for b in bboxes if b is not None)
    print(f"KCF Tracking: {success_count}/{len(bboxes)} frames successful")


# =============================================================================
# 4. MULTI-OBJECT TRACKING
# =============================================================================
print("\n--- 4. Multi-Object Tracking ---")

multi_tracking_info = """
Multi-Object Tracking (MOT):

Using cv2.MultiTracker (legacy) or manual approach:

# Legacy approach (if available)
multiTracker = cv2.legacy.MultiTracker_create()
for bbox in bboxes:
    multiTracker.add(cv2.legacy.TrackerKCF_create(), frame, bbox)
success, boxes = multiTracker.update(frame)

# Manual approach (recommended)
trackers = []
for bbox in bboxes:
    t = cv2.TrackerKCF_create()
    t.init(frame, bbox)
    trackers.append(t)

# Update each tracker
for tracker in trackers:
    success, bbox = tracker.update(frame)
"""
print(multi_tracking_info)


# =============================================================================
# 5. TRACKER COMPARISON
# =============================================================================
print("\n--- 5. Tracker Comparison ---")

comparison = """
Tracker Comparison:

| Tracker    | Speed   | Accuracy | Occlusion | Re-detection |
|------------|---------|----------|-----------|--------------|
| BOOSTING   | Slow    | Low      | Poor      | No           |
| MIL        | Slow    | Medium   | Poor      | No           |
| KCF        | Fast    | Medium   | Poor      | No           |
| TLD        | Medium  | Medium   | Good      | Yes          |
| MEDIANFLOW | Fast    | High     | Poor      | No           |
| MOSSE      | V.Fast  | Low      | Poor      | No           |
| CSRT       | Medium  | High     | Medium    | No           |
| GOTURN     | Slow    | High     | Good      | No (DNN)     |
| DaSiamRPN  | Medium  | V.High   | Good      | No (DNN)     |

Recommendations:
- Speed priority: MOSSE, KCF
- Accuracy priority: CSRT, DaSiamRPN
- Occlusion handling: TLD, GOTURN
- Real-time + accuracy: KCF or CSRT
"""
print(comparison)


# =============================================================================
# 6. DEEP LEARNING TRACKERS
# =============================================================================
print("\n--- 6. Deep Learning Trackers ---")

dl_trackers = """
Deep Learning Based Trackers:

1. GOTURN:
   - Uses CNN for tracking
   - Needs model files: goturn.caffemodel, goturn.prototxt
   - Robust to appearance changes

2. DaSiamRPN:
   - Siamese network
   - Very accurate
   - Needs model file

3. Nano Tracker:
   - Lightweight DNN
   - Mobile-friendly
   - cv2.TrackerNano_create()

Alternative Libraries:
- DeepSORT: Detection + tracking with re-ID
- ByteTrack: Simple, high-performance MOT
- SORT: Simple online realtime tracking
- Norfair: Flexible tracking library
"""
print(dl_trackers)


# =============================================================================
# 7. PRACTICAL TIPS
# =============================================================================
print("\n--- 7. Practical Tips ---")

tips = """
Tracking Best Practices:

1. Initialization:
   - Use detector for initial bbox
   - Ensure bbox is tight around object
   - Consider aspect ratio constraints

2. Handling Failures:
   - Detect tracking loss (low confidence)
   - Re-initialize with detector
   - Implement recovery mechanism

3. Performance:
   - Resize frames for speed
   - Skip frames if needed
   - Use appropriate tracker for hardware

4. Robustness:
   - Combine with detection
   - Use motion prediction (Kalman filter)
   - Handle occlusion explicitly

5. Multi-Object:
   - Use detection + tracking hybrid
   - Implement data association (Hungarian algorithm)
   - Track IDs across occlusions
"""
print(tips)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display tracking demo."""
    if not available_trackers:
        print("No trackers available. Install opencv-contrib-python")
        return

    print("\nTracking Demo")
    print("Press 'q' to quit, 'r' to restart")

    # Create tracker
    tracker = cv2.TrackerKCF_create()
    bbox = (75, 125, 50, 50)
    tracker.init(frames[0], bbox)

    frame_idx = 0
    while True:
        frame = frames[frame_idx].copy()

        if frame_idx == 0:
            # Draw initial bbox
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Initial", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "Tracking", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Lost!", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("Tracking Demo", frame)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Restart tracking
            tracker = cv2.TrackerKCF_create()
            bbox = (75, 125, 50, 50)
            tracker.init(frames[0], bbox)
            frame_idx = 0
            continue

        frame_idx = (frame_idx + 1) % len(frames)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running tracking demonstrations...")
    print("=" * 60)
    show_demo()
