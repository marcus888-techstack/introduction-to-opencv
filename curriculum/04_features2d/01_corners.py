"""
Module 4: Features2D - Corner Detection
=======================================
Detecting corners and key points in images.

Official Docs: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html

Topics Covered:
1. Harris Corner Detection
2. Shi-Tomasi (Good Features to Track)
3. FAST Corner Detection
4. Corner Subpixel Accuracy
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 4: Corner Detection")
print("=" * 60)


def create_test_image():
    """Create image with corners for testing."""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    # Draw shapes with clear corners
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 80), (350, 180), (200, 200, 200), -1)

    # Triangle
    pts = np.array([[400, 50], [350, 150], [450, 150]], np.int32)
    cv2.fillPoly(img, [pts], (180, 180, 180))

    # Checkerboard pattern
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                cv2.rectangle(img, (50 + j*40, 200 + i*40),
                             (90 + j*40, 240 + i*40), (255, 255, 255), -1)

    # Rotated rectangle
    pts = np.array([[350, 250], [400, 220], [450, 280], [400, 310]], np.int32)
    cv2.fillPoly(img, [pts], (220, 220, 220))

    return img


original = create_test_image()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)


# =============================================================================
# 1. HARRIS CORNER DETECTION
# =============================================================================
print("\n--- 1. Harris Corner Detection ---")

# Harris corner detector
# Parameters:
#   blockSize - neighborhood size (2-7)
#   ksize - Sobel kernel size (3, 5, 7)
#   k - Harris detector free parameter (0.04-0.06)

harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate for better visualization
harris = cv2.dilate(harris, None)

# Threshold to detect corners
threshold = 0.01 * harris.max()
harris_corners = harris > threshold

# Create visualization
harris_img = original.copy()
harris_img[harris_corners] = [0, 0, 255]  # Mark corners in red

# Count corners
num_corners = np.sum(harris_corners)
print(f"Harris corners found: {num_corners}")

print("Harris detector: Good for detecting corners based on gradient changes")


# =============================================================================
# 2. SHI-TOMASI CORNER DETECTION
# =============================================================================
print("\n--- 2. Shi-Tomasi (Good Features to Track) ---")

# Shi-Tomasi is often better than Harris
# Returns N strongest corners

# Parameters:
#   maxCorners - maximum number of corners to return
#   qualityLevel - quality threshold (0-1)
#   minDistance - minimum distance between corners

corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100,
    qualityLevel=0.01,
    minDistance=10
)

# Create visualization
shi_tomasi_img = original.copy()
if corners is not None:
    corners = np.int32(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(shi_tomasi_img, (x, y), 5, (0, 255, 0), -1)

    print(f"Shi-Tomasi corners found: {len(corners)}")

print("Shi-Tomasi: Better corner selection than Harris")


# =============================================================================
# 3. FAST CORNER DETECTION
# =============================================================================
print("\n--- 3. FAST Corner Detection ---")

# FAST (Features from Accelerated Segment Test)
# Very fast, good for real-time applications

# Create FAST detector
fast = cv2.FastFeatureDetector_create()

# Detect keypoints
keypoints = fast.detect(gray, None)

# Create visualization
fast_img = original.copy()
cv2.drawKeypoints(fast_img, keypoints, fast_img, color=(255, 0, 0))

print(f"FAST corners found: {len(keypoints)}")

# FAST with non-maximum suppression disabled
fast_no_nms = cv2.FastFeatureDetector_create()
fast_no_nms.setNonmaxSuppression(False)
kp_no_nms = fast_no_nms.detect(gray, None)
print(f"FAST without NMS: {len(kp_no_nms)} corners")

# Adjust threshold
fast_threshold = cv2.FastFeatureDetector_create(threshold=20)
kp_threshold = fast_threshold.detect(gray, None)
print(f"FAST with threshold=20: {len(kp_threshold)} corners")

print("FAST: Very efficient, good for real-time tracking")


# =============================================================================
# 4. SUBPIXEL CORNER ACCURACY
# =============================================================================
print("\n--- 4. Subpixel Accuracy ---")

# Refine corner locations to subpixel accuracy
# Useful for camera calibration

if corners is not None:
    # Convert corners for cornerSubPix
    corners_float = np.float32(corners)

    # Define search window and zero zone
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Refine corners
    corners_subpix = cv2.cornerSubPix(gray, corners_float, winSize, zeroZone, criteria)

    # Show difference
    print("\nOriginal vs Subpixel corner positions (first 5):")
    for i in range(min(5, len(corners))):
        orig = corners_float[i].ravel()
        subp = corners_subpix[i].ravel()
        diff = np.sqrt((orig[0]-subp[0])**2 + (orig[1]-subp[1])**2)
        print(f"  Corner {i}: ({orig[0]:.0f},{orig[1]:.0f}) -> ({subp[0]:.2f},{subp[1]:.2f}) diff={diff:.3f}")


# =============================================================================
# 5. COMPARISON
# =============================================================================
print("\n--- 5. Algorithm Comparison ---")

comparison = """
Corner Detection Algorithms:

| Algorithm   | Speed    | Quality  | Use Case                    |
|-------------|----------|----------|-----------------------------|
| Harris      | Medium   | Good     | General corner detection    |
| Shi-Tomasi  | Medium   | Better   | Feature tracking, calibration|
| FAST        | Very Fast| Good     | Real-time applications      |

Key Parameters:
- Harris: k (0.04-0.06), blockSize, ksize
- Shi-Tomasi: qualityLevel, minDistance, maxCorners
- FAST: threshold, nonmaxSuppression
"""
print(comparison)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display corner detection demos."""

    # Create comparison display
    harris_display = harris_img.copy()
    cv2.putText(harris_display, "Harris", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    shi_display = shi_tomasi_img.copy()
    cv2.putText(shi_display, f"Shi-Tomasi ({len(corners) if corners is not None else 0} pts)",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    fast_display = fast_img.copy()
    cv2.putText(fast_display, f"FAST ({len(keypoints)} pts)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Stack displays
    row1 = np.hstack([original, harris_display])
    row2 = np.hstack([shi_display, fast_display])
    display = np.vstack([row1, row2])
    display = cv2.resize(display, (1000, 800))

    cv2.imshow("Corner Detection Comparison", display)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running corner detection demonstrations...")
    print("=" * 60)
    show_demo()
