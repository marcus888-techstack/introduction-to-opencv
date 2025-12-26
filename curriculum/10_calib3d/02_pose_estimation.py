"""
Module 7: Camera Calibration - Pose Estimation
===============================================
Estimating object pose from 2D-3D correspondences.

Official Docs: https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html

Topics Covered:
1. Pose Estimation Concepts
2. solvePnP Methods
3. Rodrigues Rotation
4. AR Cube Drawing
5. Pose Refinement
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 7: Pose Estimation")
print("=" * 60)


# =============================================================================
# 1. POSE ESTIMATION CONCEPTS
# =============================================================================
print("\n--- 1. Pose Estimation Concepts ---")

pose_concepts = """
Pose Estimation Overview:

Goal: Find the position and orientation of an object (or camera)
      given 2D-3D point correspondences.

The Problem:
  Given:
    - 3D points in object/world coordinates
    - Corresponding 2D points in image
    - Camera intrinsic matrix K

  Find:
    - Rotation R (3x3 matrix or 3x1 Rodrigues vector)
    - Translation t (3x1 vector)

This is the PnP (Perspective-n-Point) problem.

Applications:
  - Augmented Reality (overlay 3D on real world)
  - Robot navigation
  - Object tracking
  - Camera localization
  - SLAM

Minimum Points:
  - 4 points for unique solution (P4P)
  - 3 points gives up to 4 solutions (P3P)
  - More points improve accuracy (use RANSAC)
"""
print(pose_concepts)


# =============================================================================
# 2. PREPARE CALIBRATION DATA
# =============================================================================
print("\n--- 2. Prepare Test Data ---")


def create_checkerboard():
    """Create a synthetic checkerboard image."""
    rows, cols = 6, 8
    square_size = 40

    h = rows * square_size
    w = cols * square_size

    board = np.zeros((h, w), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                board[i*square_size:(i+1)*square_size,
                      j*square_size:(j+1)*square_size] = 255

    return board


def load_checkerboard():
    """Load checkerboard image."""
    # Try real samples first
    for sample in ["left01.jpg", "left02.jpg", "chessboard.png"]:
        img = get_image(sample)
        if img is not None:
            print(f"  Using sample image: {sample}")
            return img

    # Fallback to synthetic
    print("  Using synthetic checkerboard")
    board = create_checkerboard()
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


# Load image and detect corners
img = load_checkerboard()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

# Try different board sizes
board_sizes = [(9, 6), (7, 5), (8, 6)]
corners = None
board_size = None

for size in board_sizes:
    ret, corners = cv2.findChessboardCorners(gray, size, None)
    if ret:
        board_size = size
        break

if corners is not None:
    print(f"  Found checkerboard: {board_size} inner corners")
    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
else:
    print("  No checkerboard found, using synthetic corners")
    # Create synthetic corners
    board_size = (7, 5)
    corners = []
    for j in range(board_size[1]):
        for i in range(board_size[0]):
            corners.append([[50 + i * 40, 50 + j * 40]])
    corners = np.array(corners, dtype=np.float32)


# =============================================================================
# 3. SOLVEPNP METHODS
# =============================================================================
print("\n--- 3. solvePnP Methods ---")

pnp_info = """
cv2.solvePnP() Methods:

Available Algorithms:
  cv2.SOLVEPNP_ITERATIVE   - Levenberg-Marquardt optimization (default)
  cv2.SOLVEPNP_P3P         - 3-point algorithm (exactly 4 points needed)
  cv2.SOLVEPNP_AP3P        - Another P3P variant
  cv2.SOLVEPNP_EPNP        - Efficient PnP (good for many points)
  cv2.SOLVEPNP_IPPE        - Infinitesimal Plane-based Pose (planar only)
  cv2.SOLVEPNP_IPPE_SQUARE - IPPE for square markers
  cv2.SOLVEPNP_SQPNP       - SQPnP method

Usage:
  success, rvec, tvec = cv2.solvePnP(
      objectPoints,    # 3D points in object frame (Nx3)
      imagePoints,     # 2D points in image (Nx2)
      cameraMatrix,    # Intrinsic matrix K
      distCoeffs,      # Distortion coefficients
      flags=cv2.SOLVEPNP_ITERATIVE
  )

With RANSAC (for outlier rejection):
  success, rvec, tvec, inliers = cv2.solvePnPRansac(
      objectPoints, imagePoints, cameraMatrix, distCoeffs,
      reprojectionError=8.0,  # Inlier threshold in pixels
      confidence=0.99,
      flags=cv2.SOLVEPNP_ITERATIVE
  )

Output:
  rvec - Rotation vector (Rodrigues form, 3x1)
  tvec - Translation vector (3x1)
"""
print(pnp_info)


# =============================================================================
# 4. PERFORM POSE ESTIMATION
# =============================================================================
print("\n--- 4. Pose Estimation Demo ---")

# Define 3D object points (checkerboard at Z=0)
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
objp *= 30  # Scale to mm (30mm squares)

# Camera matrix (assumed or from calibration)
h, w = gray.shape
focal_length = w  # Approximate focal length
camera_matrix = np.array([
    [focal_length, 0, w/2],
    [0, focal_length, h/2],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5,), dtype=np.float32)

# Solve PnP
success, rvec, tvec = cv2.solvePnP(
    objp, corners, camera_matrix, dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

if success:
    print(f"  Pose estimation successful!")
    print(f"  Rotation vector (Rodrigues): {rvec.T}")
    print(f"  Translation vector: {tvec.T}")

    # Convert Rodrigues to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    print(f"\n  Rotation matrix:\n{R}")


# =============================================================================
# 5. RODRIGUES ROTATION
# =============================================================================
print("\n--- 5. Rodrigues Rotation ---")

rodrigues_info = """
Rodrigues Rotation Representation:

The rotation vector (rvec) encodes rotation as:
  - Direction: axis of rotation
  - Magnitude: angle of rotation (radians)

Conversion:
  # Vector to Matrix
  R, jacobian = cv2.Rodrigues(rvec)

  # Matrix to Vector
  rvec, jacobian = cv2.Rodrigues(R)

Example:
  rvec = [0.1, 0.2, 0.3]  # Rotate around axis (0.1, 0.2, 0.3)
                          # by ||rvec|| = 0.374 radians

  R = cv2.Rodrigues(rvec)[0]  # 3x3 rotation matrix

Advantages of Rodrigues:
  - Only 3 parameters (vs 9 for matrix)
  - No gimbal lock (vs Euler angles)
  - Easy interpolation
"""
print(rodrigues_info)


# =============================================================================
# 6. DRAW COORDINATE AXES
# =============================================================================
print("\n--- 6. Drawing 3D Objects ---")


def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length=50):
    """Draw 3D coordinate axes on image."""
    # Define axis points in 3D
    axis_points = np.float32([
        [0, 0, 0],       # Origin
        [length, 0, 0],  # X-axis (red)
        [0, length, 0],  # Y-axis (green)
        [0, 0, -length]  # Z-axis (blue, negative = towards camera)
    ])

    # Project to 2D
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)

    origin = tuple(imgpts[0].ravel())
    x_end = tuple(imgpts[1].ravel())
    y_end = tuple(imgpts[2].ravel())
    z_end = tuple(imgpts[3].ravel())

    # Draw axes
    result = img.copy()
    cv2.line(result, origin, x_end, (0, 0, 255), 3)  # X: Red
    cv2.line(result, origin, y_end, (0, 255, 0), 3)  # Y: Green
    cv2.line(result, origin, z_end, (255, 0, 0), 3)  # Z: Blue

    # Label axes
    cv2.putText(result, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(result, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(result, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return result


def draw_cube(img, camera_matrix, dist_coeffs, rvec, tvec, size=60):
    """Draw a 3D cube on image (AR demo)."""
    # Define cube corners in 3D (sitting on the checkerboard)
    cube_points = np.float32([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],        # Bottom face
        [0, 0, -size], [size, 0, -size], [size, size, -size], [0, size, -size]  # Top face
    ])

    # Project to 2D
    imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)

    result = img.copy()

    # Draw bottom face (green)
    bottom = imgpts[:4].reshape(-1, 2)
    cv2.drawContours(result, [bottom], -1, (0, 255, 0), 2)

    # Draw top face (red)
    top = imgpts[4:].reshape(-1, 2)
    cv2.drawContours(result, [top], -1, (0, 0, 255), 2)

    # Draw vertical edges (blue)
    for i in range(4):
        pt1 = tuple(imgpts[i].ravel())
        pt2 = tuple(imgpts[i + 4].ravel())
        cv2.line(result, pt1, pt2, (255, 0, 0), 2)

    # Fill top face with semi-transparent color
    overlay = result.copy()
    cv2.fillPoly(overlay, [top], (0, 0, 255))
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

    return result


# =============================================================================
# 7. POSE REFINEMENT
# =============================================================================
print("\n--- 7. Pose Refinement ---")

refinement_info = """
Pose Refinement Techniques:

1. solvePnPRefineLM() - Levenberg-Marquardt refinement:
   rvec, tvec = cv2.solvePnPRefineLM(
       objectPoints, imagePoints,
       cameraMatrix, distCoeffs,
       rvec, tvec  # Initial estimate
   )

2. solvePnPRefineVVS() - Virtual Visual Servoing:
   rvec, tvec = cv2.solvePnPRefineVVS(
       objectPoints, imagePoints,
       cameraMatrix, distCoeffs,
       rvec, tvec
   )

When to Refine:
  - After initial estimate from P3P/EPnP
  - When tracking (use previous frame's pose as initial)
  - For higher accuracy requirements

Reprojection Error:
  # Project 3D points to 2D
  projected, _ = cv2.projectPoints(objPoints, rvec, tvec, K, dist)

  # Calculate error
  error = np.sqrt(np.mean((imagePoints - projected)**2))
  print(f"Mean reprojection error: {error:.2f} pixels")
"""
print(refinement_info)

# Calculate reprojection error
if success:
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    error = np.sqrt(np.mean((corners - projected)**2))
    print(f"\n  Reprojection error: {error:.2f} pixels")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display pose estimation demonstrations."""

    if not success:
        print("Pose estimation failed. Cannot show demo.")
        return

    # Draw checkerboard corners
    corners_img = img.copy()
    cv2.drawChessboardCorners(corners_img, board_size, corners, True)
    cv2.putText(corners_img, "Detected Corners", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Corners", corners_img)

    # Draw coordinate axes
    axes_img = draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length=60)
    cv2.putText(axes_img, "3D Axes (X=Red, Y=Green, Z=Blue)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Pose - 3D Axes", axes_img)

    # Draw AR cube
    cube_img = draw_cube(img, camera_matrix, dist_coeffs, rvec, tvec, size=60)
    cv2.putText(cube_img, "AR Cube (Augmented Reality)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Pose - AR Cube", cube_img)

    # Combined display
    combined = draw_axes(cube_img, camera_matrix, dist_coeffs, rvec, tvec, length=90)
    cv2.putText(combined, "Combined Pose Visualization", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add pose info
    pose_text = [
        f"Rotation: [{rvec[0,0]:.2f}, {rvec[1,0]:.2f}, {rvec[2,0]:.2f}]",
        f"Translation: [{tvec[0,0]:.1f}, {tvec[1,0]:.1f}, {tvec[2,0]:.1f}]"
    ]
    for i, text in enumerate(pose_text):
        cv2.putText(combined, text, (10, combined.shape[0] - 30 + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.imshow("Combined Pose", combined)

    print("\nPose Estimation Demo:")
    print("  - Corners: Detected checkerboard pattern")
    print("  - 3D Axes: Coordinate system overlay")
    print("  - AR Cube: Simple augmented reality")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running pose estimation demonstrations...")
    print("=" * 60)
    show_demo()
