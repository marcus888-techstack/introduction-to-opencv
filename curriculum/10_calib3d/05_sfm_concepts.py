"""
Module 7: Camera Calibration - Structure from Motion (SFM)
==========================================================
Fundamental concepts of Structure from Motion and epipolar geometry.

Official Docs: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
SFM Module: https://docs.opencv.org/4.x/d8/d8c/group__sfm.html

Topics Covered:
1. SFM Pipeline Overview
2. Feature Detection & Matching
3. Fundamental Matrix
4. Essential Matrix
5. Camera Pose Recovery
6. Epipolar Geometry Visualization

References:
- Middlebury: https://vision.middlebury.edu/stereo/data/
- OpenCV SFM Blog: https://opencv.org/blog/structure-from-motion-in-opencv/
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 7: Structure from Motion (SFM)")
print("=" * 60)


# =============================================================================
# 1. SFM PIPELINE OVERVIEW
# =============================================================================
print("\n--- 1. Structure from Motion Pipeline ---")

sfm_overview = """
Structure from Motion (SFM) Pipeline:

SFM reconstructs 3D structure from 2D images taken from different viewpoints.

Pipeline Steps:
┌─────────────────────────────────────────────────────────────────────┐
│                    SFM Pipeline                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Feature Detection & Matching                                     │
│     ├─ Detect keypoints (SIFT, ORB, etc.)                           │
│     └─ Match features across images                                  │
│              ↓                                                       │
│  2. Fundamental Matrix Estimation                                    │
│     ├─ F = cv2.findFundamentalMat()                                 │
│     └─ Encodes epipolar geometry (uncalibrated)                     │
│              ↓                                                       │
│  3. Essential Matrix Computation                                     │
│     ├─ E = cv2.findEssentialMat() or E = K.T @ F @ K                │
│     └─ Encodes geometry with calibrated cameras                     │
│              ↓                                                       │
│  4. Pose Recovery                                                    │
│     ├─ R, t = cv2.recoverPose(E, pts1, pts2, K)                     │
│     └─ Get relative rotation and translation                        │
│              ↓                                                       │
│  5. Triangulation                                                    │
│     ├─ pts3D = cv2.triangulatePoints(P1, P2, pts1, pts2)            │
│     └─ Reconstruct 3D points                                        │
│              ↓                                                       │
│  6. Bundle Adjustment (Optional)                                     │
│     └─ Jointly optimize cameras and 3D points                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Key Matrices:
  F - Fundamental Matrix (3x3, rank 2, 7 DoF)
      Relates pixel coordinates: x2.T @ F @ x1 = 0

  E - Essential Matrix (3x3, rank 2, 5 DoF)
      Relates normalized coordinates: x2.T @ E @ x1 = 0
      E = [t]x @ R = K2.T @ F @ K1

  Relationship: F = K2^(-T) @ E @ K1^(-1)
"""
print(sfm_overview)


# =============================================================================
# 2. LOAD IMAGES FOR SFM DEMO
# =============================================================================
print("\n--- 2. Loading Images for SFM Demo ---")


def create_simulated_view(img, angle_deg=15):
    """
    Create a simulated second view by applying a perspective transform.
    This simulates camera rotation/movement for SFM demonstration.
    """
    h, w = img.shape[:2]

    # Define source points (original corners)
    src_pts = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ])

    # Apply slight perspective change (simulating camera rotation)
    offset = int(w * 0.08)  # 8% perspective shift
    dst_pts = np.float32([
        [offset, 0], [w - offset//2, offset//2],
        [w, h], [offset//2, h - offset//2]
    ])

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply transform
    img2 = cv2.warpPerspective(img, M, (w, h))

    return img2


def load_sfm_images():
    """Load image pair for SFM demonstration."""

    # Try building image - create simulated second view
    img1 = get_image("building.jpg")
    if img1 is not None:
        print("  Using building.jpg with simulated camera motion")
        img2 = create_simulated_view(img1)
        return img1, img2, "building"

    # Try facade image from ETH3D
    img1 = get_image("facade_view1.jpg")
    if img1 is not None:
        print("  Using facade image with simulated camera motion")
        img2 = create_simulated_view(img1)
        return img1, img2, "facade"

    # Try box images (standard feature matching benchmark)
    img1 = get_image("box.png")
    img2 = get_image("box_in_scene.png")
    if img1 is not None and img2 is not None:
        print("  Using box images (object detection scenario)")
        return img1, img2, "box"

    # Fallback: create synthetic scene with clear features
    print("  Creating synthetic scene for demonstration")
    print("  Tip: Run 'python curriculum/sample_data/download_samples.py'")

    # Create a more visually appealing synthetic scene
    img1 = np.zeros((400, 600, 3), dtype=np.uint8)

    # Add a gradient background
    for y in range(400):
        img1[y, :] = [40 + y//4, 60 + y//5, 80 + y//3]

    # Add geometric shapes as features
    # Rectangles (buildings)
    cv2.rectangle(img1, (50, 150), (150, 350), (180, 160, 140), -1)
    cv2.rectangle(img1, (180, 200), (280, 350), (160, 140, 120), -1)
    cv2.rectangle(img1, (320, 180), (450, 350), (140, 120, 100), -1)

    # Windows on buildings
    for bx, by in [(70, 180), (70, 250), (110, 180), (110, 250)]:
        cv2.rectangle(img1, (bx, by), (bx+20, by+30), (200, 220, 240), -1)

    # Add some circles (trees/objects)
    cv2.circle(img1, (500, 300), 40, (60, 120, 60), -1)
    cv2.circle(img1, (530, 280), 30, (50, 100, 50), -1)

    # Add corner features
    corners = [(100, 100), (300, 80), (500, 120), (150, 350), (400, 320)]
    for pt in corners:
        cv2.circle(img1, pt, 8, (0, 200, 255), -1)
        cv2.circle(img1, pt, 8, (0, 0, 0), 2)

    # Create second view with perspective transform
    img2 = create_simulated_view(img1)

    return img1, img2, "synthetic"


img1, img2, dataset_name = load_sfm_images()
print(f"  Image 1 shape: {img1.shape}")
print(f"  Image 2 shape: {img2.shape}")


# =============================================================================
# 3. FEATURE DETECTION AND MATCHING
# =============================================================================
print("\n--- 3. Feature Detection and Matching ---")

feature_info = """
Feature Detection for SFM:

Popular Detectors:
  - SIFT: Scale-invariant, robust, slower
  - ORB: Fast, binary descriptor, good for real-time
  - AKAZE: Good balance of speed and accuracy

Matching Strategies:
  - Brute-Force: Compare all descriptors
  - FLANN: Approximate nearest neighbor (faster)
  - Ratio Test: Filter ambiguous matches (Lowe's ratio)
"""
print(feature_info)

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

# Create feature detector (SIFT or ORB)
try:
    detector = cv2.SIFT_create()
    print("  Using SIFT detector")
except cv2.error:
    detector = cv2.ORB_create(nfeatures=1000)
    print("  Using ORB detector (SIFT not available)")

# Detect keypoints and compute descriptors
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

print(f"  Image 1: {len(kp1)} keypoints")
print(f"  Image 2: {len(kp2)} keypoints")

# Match features
if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
    # Use appropriate matcher
    if desc1.dtype == np.float32:
        # SIFT descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        # ORB descriptors
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN matching with ratio test
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Apply ratio test (Lowe's ratio)
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print(f"  Good matches after ratio test: {len(good_matches)}")
else:
    good_matches = []
    print("  No descriptors found")


# =============================================================================
# 4. FUNDAMENTAL MATRIX
# =============================================================================
print("\n--- 4. Fundamental Matrix ---")

fundamental_info = """
Fundamental Matrix (F):

Relates corresponding points in two uncalibrated images:
  x2.T @ F @ x1 = 0

Properties:
  - 3x3 matrix, rank 2
  - 7 degrees of freedom (9 - 1 scale - 1 det=0)
  - Encodes epipolar geometry

Estimation Methods:
  cv2.FM_7POINT   - 7-point algorithm (returns up to 3 solutions)
  cv2.FM_8POINT   - 8-point algorithm (normalized)
  cv2.FM_RANSAC   - RANSAC (robust to outliers)
  cv2.FM_LMEDS    - Least Median of Squares

Usage:
  F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

Epipolar Lines:
  l2 = F @ x1      (line in image 2)
  l1 = F.T @ x2    (line in image 1)
"""
print(fundamental_info)

F = None
pts1 = None
pts2 = None
inlier_mask = None

if len(good_matches) >= 8:
    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Compute Fundamental Matrix with RANSAC
    F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

    if F is not None:
        inliers = np.sum(inlier_mask)
        print(f"  Fundamental Matrix computed")
        print(f"  Inliers: {inliers}/{len(good_matches)}")
        print(f"  F matrix:\n{F}")
else:
    print(f"  Not enough matches ({len(good_matches)}). Need at least 8.")


# =============================================================================
# 5. ESSENTIAL MATRIX
# =============================================================================
print("\n--- 5. Essential Matrix ---")

essential_info = """
Essential Matrix (E):

Relates corresponding points in calibrated cameras:
  x2_norm.T @ E @ x1_norm = 0

Properties:
  - 3x3 matrix, rank 2
  - 5 degrees of freedom
  - E = [t]x @ R (encodes rotation and translation)

Relationship to F:
  E = K2.T @ F @ K1
  F = K2^(-T) @ E @ K1^(-1)

Estimation:
  E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

Decomposition:
  - Gives 4 possible (R, t) combinations
  - Only 1 has positive depth (cheirality check)
"""
print(essential_info)

# Assume camera matrix (for demo purposes)
h, w = gray1.shape
focal = w  # Approximate focal length
K = np.array([
    [focal, 0, w/2],
    [0, focal, h/2],
    [0, 0, 1]
], dtype=np.float64)

E = None
R = None
t = None

if F is not None and pts1 is not None:
    # Method 1: Compute E from F
    E_from_F = K.T @ F @ K
    print(f"  E from F:\n{E_from_F}")

    # Method 2: Compute E directly (more robust)
    E, E_mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)

    if E is not None:
        print(f"  E (direct):\n{E}")

        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)

        print(f"\n  Recovered Pose:")
        print(f"  Rotation matrix:\n{R}")
        print(f"  Translation vector: {t.T}")


# =============================================================================
# 6. EPIPOLAR GEOMETRY VISUALIZATION
# =============================================================================
print("\n--- 6. Epipolar Geometry ---")

epipolar_info = """
Epipolar Geometry:

┌─────────────────────────────────────────────────────────────────────┐
│                     Epipolar Geometry                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                        P (3D point)                                  │
│                           ●                                          │
│                          /│\\                                         │
│                         / │ \\                                        │
│                        /  │  \\                                       │
│         epipolar      /   │   \\      epipolar                        │
│         plane        /    │    \\     plane                           │
│                     /     │     \\                                    │
│              C1 ●───/─────│──────\\───● C2                            │
│                │   /      │       \\  │                               │
│            ┌──┼──/────┐   │   ┌────\\─┼──┐                            │
│            │  │ x1    │ baseline  │ x2 │ │                            │
│            │  e1      │   │   │      e2 │                            │
│            └──────────┘   │   └─────────┘                            │
│             Image 1       │    Image 2                               │
│                           │                                          │
│  e1, e2 = epipoles (projection of other camera center)              │
│  x1, x2 = corresponding points                                      │
│  Epipolar line: x2 lies on line l2 = F @ x1                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Key Insight:
  If we know x1, we know x2 lies on the epipolar line l2.
  This reduces 2D search to 1D search!
"""
print(epipolar_info)


def draw_epipolar_lines(img1, img2, pts1, pts2, F, num_lines=10):
    """Draw epipolar lines for corresponding points."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    result1 = img1.copy()
    result2 = img2.copy()

    # Select subset of points
    indices = np.random.choice(len(pts1), min(num_lines, len(pts1)), replace=False)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
              (0, 128, 255), (128, 255, 0)]

    for i, idx in enumerate(indices):
        color = colors[i % len(colors)]

        # Points
        pt1 = tuple(pts1[idx].astype(int))
        pt2 = tuple(pts2[idx].astype(int))

        # Draw points
        cv2.circle(result1, pt1, 5, color, -1)
        cv2.circle(result2, pt2, 5, color, -1)

        # Compute epipolar lines
        # Line in image 2 for point in image 1
        pt1_h = np.array([pt1[0], pt1[1], 1.0])
        l2 = F @ pt1_h
        # Line equation: ax + by + c = 0
        a, b, c = l2
        # Find intersection with image borders
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c / b)
            x1, y1 = w2, int(-(a * w2 + c) / b)
            cv2.line(result2, (x0, y0), (x1, y1), color, 1)

        # Line in image 1 for point in image 2
        pt2_h = np.array([pt2[0], pt2[1], 1.0])
        l1 = F.T @ pt2_h
        a, b, c = l1
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c / b)
            x1, y1 = w1, int(-(a * w1 + c) / b)
            cv2.line(result1, (x0, y0), (x1, y1), color, 1)

    return result1, result2


# =============================================================================
# 7. RECONSTRUCTION SUMMARY
# =============================================================================
print("\n--- 7. Reconstruction Summary ---")

summary = """
From Images to 3D:

1. Given: Two (or more) images of a scene

2. Feature Matching:
   - Detect features in each image
   - Match corresponding features

3. Geometric Estimation:
   - Estimate F or E from matches
   - Recover camera pose (R, t)

4. Triangulation:
   - Reconstruct 3D points from matches
   - P = triangulate(x1, x2, P1, P2)

5. Refinement:
   - Bundle adjustment
   - Outlier removal
   - Dense reconstruction

Tools & Libraries:
  - OpenCV (basic SFM, stereo)
  - COLMAP (state-of-the-art SFM + MVS)
  - OpenMVG/OpenMVS (open source pipeline)
  - Meshroom (photogrammetry GUI)
"""
print(summary)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display SFM demonstrations."""

    # Feature matches
    if len(good_matches) > 0:
        matches_img = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.putText(matches_img, f"Feature Matches ({len(good_matches)} total)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("SFM: Feature Matches", matches_img)

    # Epipolar lines
    if F is not None and pts1 is not None and inlier_mask is not None:
        # Filter inliers
        inlier_pts1 = pts1[inlier_mask.ravel() == 1]
        inlier_pts2 = pts2[inlier_mask.ravel() == 1]

        if len(inlier_pts1) > 0:
            epi1, epi2 = draw_epipolar_lines(img1, img2, inlier_pts1, inlier_pts2, F)
            epipolar_display = np.hstack([epi1, epi2])
            cv2.putText(epipolar_display, "Epipolar Lines (points should lie on lines)", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("SFM: Epipolar Geometry", epipolar_display)

    # Camera pose visualization (simple)
    if R is not None and t is not None:
        pose_img = np.zeros((400, 600, 3), dtype=np.uint8)

        # Draw camera 1 (at origin)
        cam1_pos = (200, 300)
        cv2.rectangle(pose_img, (cam1_pos[0]-20, cam1_pos[1]-15),
                     (cam1_pos[0]+20, cam1_pos[1]+15), (0, 255, 0), 2)
        cv2.putText(pose_img, "Cam1", (cam1_pos[0]-15, cam1_pos[1]+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw camera 2 (relative position)
        # Scale translation for visualization
        scale = 100
        cam2_offset = (int(t[0, 0] * scale), int(-t[2, 0] * scale))  # x, z plane
        cam2_pos = (cam1_pos[0] + cam2_offset[0], cam1_pos[1] + cam2_offset[1])
        cv2.rectangle(pose_img, (cam2_pos[0]-20, cam2_pos[1]-15),
                     (cam2_pos[0]+20, cam2_pos[1]+15), (0, 0, 255), 2)
        cv2.putText(pose_img, "Cam2", (cam2_pos[0]-15, cam2_pos[1]+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw baseline
        cv2.line(pose_img, cam1_pos, cam2_pos, (255, 255, 0), 2)

        cv2.putText(pose_img, "Recovered Camera Pose (top view)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(pose_img, f"Translation: [{t[0,0]:.2f}, {t[1,0]:.2f}, {t[2,0]:.2f}]",
                   (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("SFM: Camera Pose", pose_img)

    print(f"\nDataset: {dataset_name}")
    print(f"Matches: {len(good_matches)}")
    if F is not None:
        print(f"Fundamental matrix computed successfully")
    if R is not None:
        print(f"Camera pose recovered successfully")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running SFM demonstrations...")
    print("=" * 60)
    show_demo()
