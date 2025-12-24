"""
Module 6: Video Analysis - Background Subtraction
==================================================
Foreground detection and background modeling.

Official Docs: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

Topics Covered:
1. MOG2 Background Subtractor
2. KNN Background Subtractor
3. Foreground Mask Processing
4. Shadow Detection
5. Parameters Tuning
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 6: Background Subtraction")
print("=" * 60)


def create_video_frames(num_frames=60):
    """Generate synthetic video frames with moving objects."""
    frames = []

    for i in range(num_frames):
        # Background (static)
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:150] = (100, 80, 60)   # Sky
        frame[150:] = (60, 100, 60)    # Ground

        # Static elements
        cv2.rectangle(frame, (50, 100), (100, 150), (80, 80, 80), -1)  # House
        cv2.rectangle(frame, (300, 80), (350, 150), (80, 80, 80), -1)  # House

        # Moving object 1: Circle moving right
        x1 = 50 + int(300 * i / num_frames)
        cv2.circle(frame, (x1, 200), 20, (0, 0, 255), -1)

        # Moving object 2: Rectangle moving left
        x2 = 350 - int(250 * i / num_frames)
        cv2.rectangle(frame, (x2, 170), (x2+40, 220), (255, 0, 0), -1)

        # Moving object 3: Smaller circle with different speed
        x3 = int(50 + 150 * np.sin(i * 0.1) + 150)
        cv2.circle(frame, (x3, 250), 15, (0, 255, 0), -1)

        frames.append(frame)

    return frames


frames = create_video_frames()
print(f"Generated {len(frames)} test frames")


# =============================================================================
# 1. MOG2 BACKGROUND SUBTRACTOR
# =============================================================================
print("\n--- 1. MOG2 Background Subtractor ---")

# Create MOG2 subtractor
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,           # Number of frames for background model
    varThreshold=16,       # Threshold for foreground detection
    detectShadows=True     # Detect shadows (shown in gray)
)

print("MOG2: Gaussian Mixture-based background/foreground segmentation")

# Process frames
mog2_results = []
for frame in frames:
    mask = mog2.apply(frame)
    mog2_results.append(mask)

# Get learned background
mog2_bg = mog2.getBackgroundImage()
print(f"Background model shape: {mog2_bg.shape if mog2_bg is not None else 'None'}")


# =============================================================================
# 2. KNN BACKGROUND SUBTRACTOR
# =============================================================================
print("\n--- 2. KNN Background Subtractor ---")

# Create KNN subtractor
knn = cv2.createBackgroundSubtractorKNN(
    history=500,           # Number of frames
    dist2Threshold=400.0,  # Threshold on squared distance
    detectShadows=True     # Detect shadows
)

print("KNN: K-nearest neighbors based segmentation")

# Process frames
knn_results = []
for frame in frames:
    mask = knn.apply(frame)
    knn_results.append(mask)

# Get learned background
knn_bg = knn.getBackgroundImage()


# =============================================================================
# 3. MASK PROCESSING
# =============================================================================
print("\n--- 3. Mask Processing ---")

# Get last mask as example
mask = mog2_results[-1].copy()

# Original mask values:
# 0 = background
# 127 = shadow (if detectShadows=True)
# 255 = foreground

# Remove shadows (keep only definite foreground)
mask_no_shadow = np.where(mask == 255, 255, 0).astype(np.uint8)

# Morphological operations to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Opening removes small noise
mask_opened = cv2.morphologyEx(mask_no_shadow, cv2.MORPH_OPEN, kernel)

# Closing fills small holes
mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

# Dilation expands foreground
mask_dilated = cv2.dilate(mask_closed, kernel, iterations=2)

print("Mask processing steps: shadow removal -> opening -> closing -> dilation")


# =============================================================================
# 4. FOREGROUND DETECTION
# =============================================================================
print("\n--- 4. Foreground Detection ---")

# Find contours in mask
contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

# Filter by area
min_area = 200
detected_objects = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        detected_objects.append({
            'bbox': (x, y, w, h),
            'area': area,
            'centroid': (x + w//2, y + h//2)
        })

print(f"Detected {len(detected_objects)} moving objects")

# Visualize detections
detection_img = frames[-1].copy()
for obj in detected_objects:
    x, y, w, h = obj['bbox']
    cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.circle(detection_img, obj['centroid'], 3, (0, 0, 255), -1)


# =============================================================================
# 5. PARAMETER TUNING
# =============================================================================
print("\n--- 5. Parameter Tuning ---")

params_info = """
MOG2 Parameters:
  history        - Frames for background model (higher = slower adaptation)
  varThreshold   - Foreground threshold (lower = more sensitive)
  detectShadows  - Enable shadow detection
  shadowThreshold - Shadow detection threshold

KNN Parameters:
  history        - Frames for background model
  dist2Threshold - Distance threshold (lower = more sensitive)
  detectShadows  - Enable shadow detection

Runtime adjustments:
  setHistory(n)           - Update history length
  setVarThreshold(t)      - Update threshold
  setDetectShadows(bool)  - Toggle shadow detection

Tips:
  - Increase history for stable backgrounds
  - Decrease varThreshold for more sensitivity
  - Use morphology to clean masks
"""
print(params_info)


# =============================================================================
# 6. COMPARISON
# =============================================================================
print("\n--- 6. Algorithm Comparison ---")

comparison = """
Background Subtraction Comparison:

| Algorithm | Speed  | Accuracy | Best For                     |
|-----------|--------|----------|------------------------------|
| MOG2      | Fast   | Good     | General purpose              |
| KNN       | Medium | Better   | More complex backgrounds     |

MOG2:
- Uses Gaussian Mixture Model
- Faster computation
- Good for simple scenes

KNN:
- Uses K-nearest neighbors
- Better at handling shadows
- Handles multi-modal backgrounds

Both support:
- Shadow detection
- Dynamic background learning
- Online adaptation
"""
print(comparison)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display background subtraction demos."""
    print("\n" + "=" * 60)
    print("Background Subtraction Demo")
    print("Press 'q' to quit, 'space' to pause/resume")
    print("=" * 60)

    # Recreate subtractors for fresh demo
    mog2_demo = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    knn_demo = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            frame = frames[frame_idx]

            # Apply subtractors
            mog2_mask = mog2_demo.apply(frame)
            knn_mask = knn_demo.apply(frame)

            # Get backgrounds
            mog2_bg = mog2_demo.getBackgroundImage()
            knn_bg = knn_demo.getBackgroundImage()

            # Process mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mog2_clean = cv2.morphologyEx(mog2_mask, cv2.MORPH_OPEN, kernel)
            mog2_clean = cv2.morphologyEx(mog2_clean, cv2.MORPH_CLOSE, kernel)

            # Convert masks to BGR for display
            mog2_display = cv2.cvtColor(mog2_mask, cv2.COLOR_GRAY2BGR)
            knn_display = cv2.cvtColor(knn_mask, cv2.COLOR_GRAY2BGR)
            clean_display = cv2.cvtColor(mog2_clean, cv2.COLOR_GRAY2BGR)

            # Draw detections
            contours, _ = cv2.findContours(mog2_clean, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            detection = frame.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(detection, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Create display grid
            row1 = np.hstack([frame, detection])
            row2 = np.hstack([mog2_display, clean_display])

            # Add labels
            cv2.putText(row1, "Original", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(row1, "Detections", (410, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(row2, "MOG2 Mask", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(row2, "Cleaned Mask", (410, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            display = np.vstack([row1, row2])
            cv2.imshow("Background Subtraction", display)

            # Show background model
            if mog2_bg is not None:
                cv2.imshow("Learned Background", mog2_bg)

            frame_idx = (frame_idx + 1) % len(frames)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_demo()
