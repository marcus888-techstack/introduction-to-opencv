"""
Module 6: Video Analysis - Optical Flow
========================================
Tracking motion between frames.

Official Docs: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html

Topics Covered:
1. Lucas-Kanade Optical Flow (Sparse)
2. Dense Optical Flow (Farneback)
3. Feature Tracking
4. Motion Visualization
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_video

print("=" * 60)
print("Module 6: Optical Flow")
print("=" * 60)


def load_video_frames():
    """Load two consecutive frames from a real video, or create synthetic frames."""
    # Try to load sample video
    for video_name in ["vtest.avi", "slow_traffic_small.mp4"]:
        video_path = get_video(video_name)
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret1, frame1 = cap.read()
                # Skip a few frames for visible motion
                for _ in range(5):
                    cap.read()
                ret2, frame2 = cap.read()
                cap.release()

                if ret1 and ret2:
                    print(f"Using sample video: {video_name}")
                    # Resize for consistent display
                    frame1 = cv2.resize(frame1, (600, 400))
                    frame2 = cv2.resize(frame2, (600, 400))
                    return frame1, frame2

    # Fallback to synthetic
    print("No sample video found. Using synthetic frames.")
    print("Run: python curriculum/sample_data/download_samples.py")
    return create_test_frames()


def create_test_frames():
    """Create two frames with motion for testing (fallback)."""
    # Frame 1
    frame1 = np.zeros((400, 600, 3), dtype=np.uint8)
    frame1[:] = (50, 50, 50)

    # Moving objects at position 1
    cv2.circle(frame1, (100, 200), 40, (0, 255, 0), -1)
    cv2.rectangle(frame1, (300, 150), (400, 250), (255, 0, 0), -1)
    cv2.circle(frame1, (500, 100), 30, (0, 255, 255), -1)

    # Frame 2 (objects moved)
    frame2 = np.zeros((400, 600, 3), dtype=np.uint8)
    frame2[:] = (50, 50, 50)

    # Objects at new positions
    cv2.circle(frame2, (130, 210), 40, (0, 255, 0), -1)  # Moved right and down
    cv2.rectangle(frame2, (280, 160), (380, 260), (255, 0, 0), -1)  # Moved left and down
    cv2.circle(frame2, (520, 130), 30, (0, 255, 255), -1)  # Moved right and down

    return frame1, frame2


frame1, frame2 = load_video_frames()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


# =============================================================================
# 1. LUCAS-KANADE OPTICAL FLOW (SPARSE)
# =============================================================================
print("\n--- 1. Lucas-Kanade Optical Flow ---")

# Find good features to track in first frame
corners = cv2.goodFeaturesToTrack(
    gray1,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

print(f"Features to track: {len(corners) if corners is not None else 0}")

# Parameters for Lucas-Kanade
lk_params = dict(
    winSize=(15, 15),           # Window size for each pyramid level
    maxLevel=2,                  # Pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Calculate optical flow
if corners is not None:
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, corners, None, **lk_params
    )

    # Select good points
    good_old = corners[status == 1]
    good_new = next_pts[status == 1]

    print(f"Successfully tracked: {len(good_new)}")

    # Draw tracks
    lk_img = frame2.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)

        # Draw line from old to new position
        cv2.line(lk_img, (a, b), (c, d), (0, 255, 255), 2)
        # Draw current position
        cv2.circle(lk_img, (a, b), 5, (0, 0, 255), -1)

lk_params_info = """
Lucas-Kanade Parameters:
  winSize    - Search window size (larger = smoother but less precise)
  maxLevel   - Pyramid levels (higher = handles larger motions)
  criteria   - Termination criteria for iterative search
"""
print(lk_params_info)


# =============================================================================
# 2. DENSE OPTICAL FLOW (FARNEBACK)
# =============================================================================
print("\n--- 2. Dense Optical Flow (Farneback) ---")

# Calculate dense optical flow
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    pyr_scale=0.5,      # Pyramid scale
    levels=3,            # Pyramid levels
    winsize=15,          # Averaging window
    iterations=3,        # Iterations per level
    poly_n=5,            # Polynomial expansion neighborhood
    poly_sigma=1.2,      # Gaussian std for polynomial
    flags=0
)

print(f"Flow shape: {flow.shape}")  # (H, W, 2) for dx, dy

# Visualize flow as color
def flow_to_color(flow):
    """Convert optical flow to HSV color representation."""
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Saturation

    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Hue represents direction
    hsv[..., 0] = ang * 180 / np.pi / 2

    # Value represents magnitude
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


flow_color = flow_to_color(flow)

# Draw flow vectors (sampled)
flow_arrows = frame2.copy()
step = 20
for y in range(0, flow.shape[0], step):
    for x in range(0, flow.shape[1], step):
        dx, dy = flow[y, x]
        if np.sqrt(dx**2 + dy**2) > 1:  # Only show significant motion
            cv2.arrowedLine(flow_arrows, (x, y),
                           (int(x + dx*2), int(y + dy*2)),
                           (0, 255, 0), 1, tipLength=0.3)

farneback_info = """
Farneback Parameters:
  pyr_scale  - Pyramid scale (0.5 = half)
  levels     - Pyramid levels
  winsize    - Averaging window
  iterations - Iterations per level
  poly_n     - Polynomial neighborhood (5 or 7)
  poly_sigma - Gaussian smoothing
"""
print(farneback_info)


# =============================================================================
# 3. MOTION DETECTION
# =============================================================================
print("\n--- 3. Motion Detection from Flow ---")

# Get motion magnitude
magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

# Threshold for motion
motion_threshold = 2.0
motion_mask = (magnitude > motion_threshold).astype(np.uint8) * 255

# Find contours of moving regions
contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

motion_img = frame2.copy()
for cnt in contours:
    if cv2.contourArea(cnt) > 100:  # Filter small regions
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(motion_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

print(f"Motion regions detected: {len(contours)}")


# =============================================================================
# 4. FLOW STATISTICS
# =============================================================================
print("\n--- 4. Flow Statistics ---")

# Average motion
avg_dx = np.mean(flow[..., 0])
avg_dy = np.mean(flow[..., 1])
avg_mag = np.mean(magnitude)

print(f"Average horizontal motion: {avg_dx:.2f} px")
print(f"Average vertical motion: {avg_dy:.2f} px")
print(f"Average magnitude: {avg_mag:.2f} px")

# Direction histogram
angles = np.arctan2(flow[..., 1], flow[..., 0])
angles_deg = angles * 180 / np.pi

# Dominant direction
hist, bins = np.histogram(angles_deg[magnitude > 1].flatten(), bins=8, range=(-180, 180))
dominant_dir = bins[np.argmax(hist)]
print(f"Dominant motion direction: {dominant_dir:.0f} degrees")


# =============================================================================
# 5. COMPARISON
# =============================================================================
print("\n--- 5. Algorithm Comparison ---")

comparison = """
Optical Flow Methods:

| Method          | Type   | Speed  | Best For                    |
|-----------------|--------|--------|------------------------------|
| Lucas-Kanade    | Sparse | Fast   | Feature tracking, small motion|
| Farneback       | Dense  | Medium | Full motion field            |
| DualTVL1        | Dense  | Slow   | High-quality motion          |
| RLOF            | Sparse | Fast   | Robust tracking              |

Sparse vs Dense:
- Sparse: Track specific points, faster, needs good features
- Dense: Motion at every pixel, slower, complete motion field

Use Cases:
- Object tracking: Lucas-Kanade
- Motion visualization: Farneback
- Video stabilization: Dense flow
- Gesture recognition: Dense flow
"""
print(comparison)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display optical flow demos."""

    # Original frames
    frames = np.hstack([frame1, frame2])
    cv2.putText(frames, "Frame 1", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frames, "Frame 2", (610, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Input Frames", frames)

    # Lucas-Kanade result
    if corners is not None:
        cv2.putText(lk_img, "Lucas-Kanade Tracks", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Lucas-Kanade Optical Flow", lk_img)

    # Dense flow visualization
    cv2.putText(flow_color, "Dense Flow (Color)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Farneback Optical Flow", flow_color)

    # Flow arrows
    cv2.putText(flow_arrows, "Flow Vectors", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Flow Arrows", flow_arrows)

    # Motion detection
    cv2.putText(motion_img, "Motion Detection", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Motion Detection", motion_img)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running optical flow demonstrations...")
    print("=" * 60)
    show_demo()
