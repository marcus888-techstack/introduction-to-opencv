"""
Module 7: Camera Calibration - Basics
=====================================
Camera calibration and undistortion.

Official Docs: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

Topics Covered:
1. Calibration Pattern Detection
2. Camera Calibration
3. Undistortion
4. Perspective Transform
5. Homography
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 7: Camera Calibration")
print("=" * 60)


# =============================================================================
# 1. CALIBRATION CONCEPTS
# =============================================================================
print("\n--- 1. Camera Calibration Concepts ---")

concepts = """
Camera Parameters:

Intrinsic Parameters (camera matrix K):
  fx, fy - Focal lengths in pixels
  cx, cy - Principal point (optical center)

  K = | fx  0  cx |
      | 0  fy  cy |
      | 0   0   1 |

Distortion Coefficients:
  k1, k2, k3 - Radial distortion
  p1, p2     - Tangential distortion

  distCoeffs = [k1, k2, p1, p2, k3]

Extrinsic Parameters:
  R - Rotation matrix (3x3)
  t - Translation vector (3x1)
"""
print(concepts)


# =============================================================================
# 2. CREATE SYNTHETIC CHECKERBOARD
# =============================================================================
print("\n--- 2. Creating Test Pattern ---")


def create_checkerboard(rows=6, cols=8, square_size=40):
    """Create a synthetic checkerboard image."""
    h = rows * square_size
    w = cols * square_size

    board = np.zeros((h, w), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                board[i*square_size:(i+1)*square_size,
                      j*square_size:(j+1)*square_size] = 255

    return board


# Create checkerboard
board_size = (7, 5)  # Inner corners (cols-1, rows-1)
checkerboard = create_checkerboard(rows=6, cols=8)

# Add some distortion to simulate camera image
h, w = checkerboard.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 15, 0.9)  # Rotate and scale
distorted = cv2.warpAffine(checkerboard, M, (w + 100, h + 100),
                           borderValue=128)

print(f"Checkerboard size: {checkerboard.shape}")
print(f"Inner corners: {board_size}")


# =============================================================================
# 3. FINDING CORNERS
# =============================================================================
print("\n--- 3. Finding Checkerboard Corners ---")

# Find chessboard corners
ret, corners = cv2.findChessboardCorners(checkerboard, board_size, None)

print(f"Corners found: {ret}")
if ret:
    print(f"Number of corners: {len(corners)}")

    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(checkerboard, corners, (11, 11),
                                        (-1, -1), criteria)

    # Draw corners
    corners_img = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(corners_img, board_size, corners_refined, ret)


# =============================================================================
# 4. CALIBRATION PROCESS
# =============================================================================
print("\n--- 4. Camera Calibration Process ---")

calibration_steps = """
Calibration Steps:

1. Prepare object points (3D):
   - Define real-world coordinates of pattern
   - Usually in millimeters or any unit

2. Collect image points (2D):
   - Detect pattern in multiple images
   - Different angles and positions

3. Calibrate camera:
   - cv2.calibrateCamera()
   - Returns camera matrix and distortion coefficients

4. Evaluate calibration:
   - Calculate reprojection error
   - Should be < 1 pixel ideally
"""
print(calibration_steps)

# Prepare object points
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

# For demo, we'll use single image (in practice, use 10-20 images)
objpoints = [objp]  # 3D points
imgpoints = [corners_refined] if ret else []  # 2D points

if imgpoints:
    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, checkerboard.shape[::-1], None, None
    )

    print(f"\nCalibration successful: {ret}")
    print(f"Camera Matrix:\n{mtx}")
    print(f"\nDistortion Coefficients: {dist.ravel()}")

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                          mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"Mean reprojection error: {mean_error/len(objpoints):.4f} pixels")


# =============================================================================
# 5. UNDISTORTION
# =============================================================================
print("\n--- 5. Image Undistortion ---")

# Create a sample distorted image
def create_distorted_grid():
    """Create a grid with barrel distortion effect."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Draw grid
    for i in range(0, 400, 20):
        cv2.line(img, (i, 0), (i, 400), (100, 100, 100), 1)
        cv2.line(img, (0, i), (400, i), (100, 100, 100), 1)

    return img


grid = create_distorted_grid()

# Simulate distortion with a simple radial effect
# In practice, use cv2.undistort with real distortion coefficients

undistort_info = """
Undistortion Methods:

1. cv2.undistort():
   - Simple one-step undistortion
   - dst = cv2.undistort(src, mtx, dist)

2. cv2.initUndistortRectifyMap() + cv2.remap():
   - Two-step, but faster for multiple images
   - Pre-compute maps once, apply to many images

Example:
  mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
  dst = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)
"""
print(undistort_info)


# =============================================================================
# 6. PERSPECTIVE TRANSFORM
# =============================================================================
print("\n--- 6. Perspective Transform ---")

# Create test image with quadrilateral
test_img = np.zeros((400, 400, 3), dtype=np.uint8)
pts = np.array([[80, 80], [320, 100], [350, 350], [50, 300]], np.int32)
cv2.fillPoly(test_img, [pts], (0, 255, 0))
cv2.putText(test_img, "Skewed", (120, 200), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2)

# Define source and destination points
src_pts = np.float32([[80, 80], [320, 100], [350, 350], [50, 300]])
dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

# Get perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
print(f"Perspective Matrix:\n{M}")

# Apply transform
warped = cv2.warpPerspective(test_img, M, (300, 300))


# =============================================================================
# 7. HOMOGRAPHY
# =============================================================================
print("\n--- 7. Finding Homography ---")

# Homography maps points from one plane to another
homography_info = """
Homography:

A 3x3 transformation matrix that maps points between two planes:
  dst = H * src

Uses:
  - Image stitching
  - Perspective correction
  - Augmented reality
  - Object pose estimation

cv2.findHomography():
  - Uses RANSAC to handle outliers
  - Needs at least 4 point correspondences

Example:
  H, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
"""
print(homography_info)

# Example: Find homography from corresponding points
src_points = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]]).reshape(-1, 1, 2)
dst_points = np.float32([[10, 20], [90, 10], [95, 105], [5, 95]]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
print(f"\nHomography matrix:\n{H}")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display calibration demos."""

    # Checkerboard with corners
    if ret:
        cv2.imshow("Checkerboard Corners", corners_img)

    # Original checkerboard
    cv2.imshow("Original Checkerboard", checkerboard)

    # Perspective transform
    transform_display = np.hstack([
        cv2.resize(test_img, (300, 300)),
        warped
    ])
    cv2.putText(transform_display, "Original", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(transform_display, "Warped", (310, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Perspective Transform", transform_display)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running camera calibration demonstrations...")
    print("=" * 60)
    show_demo()
