"""
Module 7: Camera Calibration - 3D Reconstruction
=================================================
Triangulation and 3D point cloud generation from real stereo images.

Official Docs: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

Topics Covered:
1. Triangulation Theory
2. Stereo to 3D Point Cloud
3. Feature-based Sparse Reconstruction
4. Dense Reconstruction
5. Point Cloud Export (PLY format)

Uses Middlebury Stereo Benchmark images.
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 7: 3D Reconstruction")
print("=" * 60)


# =============================================================================
# 1. TRIANGULATION CONCEPTS
# =============================================================================
print("\n--- 1. Triangulation Concepts ---")

triangulation_concepts = """
Triangulation Overview:

Given:
  - Corresponding 2D points in two or more images
  - Camera matrices (intrinsic K and extrinsic [R|t])

Find:
  - 3D coordinates of the points

How It Works:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       Triangulation                                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │                           ● P (3D point)                             │
  │                          ╱ ╲                                         │
  │                         ╱   ╲                                        │
  │                  ray 1 ╱     ╲ ray 2                                 │
  │                       ╱       ╲                                      │
  │            Camera 1 ●─────────● Camera 2                            │
  │                     │         │                                      │
  │                  ┌──┼──┐   ┌──┼──┐                                  │
  │                  │p1●  │   │  ●p2│                                  │
  │                  └─────┘   └─────┘                                  │
  │                                                                      │
  │   Rays from cameras through image points should intersect at P      │
  │   In practice, rays don't exactly meet → minimize distance          │
  │                                                                      │
  └─────────────────────────────────────────────────────────────────────┘

Key Equation:
  depth = (focal_length × baseline) / disparity

  Where:
    - baseline = distance between camera centers
    - disparity = x_left - x_right (horizontal shift)
"""
print(triangulation_concepts)


# =============================================================================
# 2. LOAD REAL STEREO IMAGES
# =============================================================================
print("\n--- 2. Loading Real Stereo Images ---")


def load_stereo_pair():
    """Load stereo image pair from Middlebury benchmark."""

    # Try Middlebury datasets in order of preference
    datasets = [
        ("cones_left.png", "cones_right.png", "Cones"),
        ("teddy_left.png", "teddy_right.png", "Teddy"),
        ("tsukuba_left.ppm", "tsukuba_right.ppm", "Tsukuba"),
    ]

    for left_name, right_name, dataset_name in datasets:
        left = get_image(left_name)
        right = get_image(right_name)

        if left is not None and right is not None:
            print(f"  Loaded {dataset_name} stereo pair from Middlebury benchmark")
            print(f"  Left image: {left.shape}")
            print(f"  Right image: {right.shape}")
            return left, right, dataset_name

    # Try OpenCV calibration images as fallback
    left = get_image("left01.jpg")
    right = get_image("right01.jpg")

    if left is not None and right is not None:
        print("  Using OpenCV calibration stereo pair")
        return left, right, "OpenCV"

    # Create synthetic stereo pair
    print("  Creating synthetic stereo pair for demonstration")
    print("  Tip: Run 'python curriculum/sample_data/download_samples.py'")

    # Create a synthetic scene
    left = np.zeros((400, 500, 3), dtype=np.uint8)

    # Background gradient
    for y in range(400):
        left[y, :] = [60 + y//8, 80 + y//6, 100 + y//5]

    # Add objects at different depths
    # Far object (small disparity)
    cv2.rectangle(left, (350, 100), (450, 200), (180, 140, 100), -1)
    cv2.rectangle(left, (370, 120), (390, 150), (220, 200, 180), -1)

    # Middle object
    cv2.rectangle(left, (150, 150), (280, 320), (100, 120, 160), -1)
    cv2.rectangle(left, (170, 170), (200, 220), (180, 200, 220), -1)
    cv2.rectangle(left, (220, 170), (250, 220), (180, 200, 220), -1)

    # Near object (large disparity)
    cv2.circle(left, (80, 280), 50, (60, 100, 60), -1)

    # Create right image with disparity shift
    right = np.zeros_like(left)
    # Shift objects by different amounts based on "depth"
    # Far objects shift less, near objects shift more
    right[:, 5:] = left[:, :-5]  # Base shift

    return left, right, "Synthetic"


# Load stereo images
left_img, right_img, dataset_name = load_stereo_pair()

# Convert to grayscale for stereo matching
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img


# =============================================================================
# 3. COMPUTE DISPARITY MAP
# =============================================================================
print("\n--- 3. Computing Disparity Map ---")

disparity_info = """
Disparity Calculation:

disparity[y,x] = x_left - x_right

For a point visible in both images:
  - If close to camera: large disparity (pixels shift a lot)
  - If far from camera: small disparity (pixels shift little)

Converting disparity to depth:
  Z = (f × B) / d

  Where:
    Z = depth
    f = focal length (pixels)
    B = baseline (distance between cameras)
    d = disparity
"""
print(disparity_info)

# Create stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 6,  # Must be divisible by 16
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute disparity
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

print(f"  Disparity map shape: {disparity.shape}")
print(f"  Disparity range: [{disparity.min():.1f}, {disparity.max():.1f}]")


# =============================================================================
# 4. 3D POINT CLOUD GENERATION
# =============================================================================
print("\n--- 4. Generating 3D Point Cloud ---")

pointcloud_info = """
Point Cloud from Disparity:

Method 1: Using Q matrix (from stereo calibration)
  points_3d = cv2.reprojectImageTo3D(disparity, Q)

Method 2: Manual calculation
  For each pixel (x, y) with disparity d:
    X = (x - cx) × Z / f
    Y = (y - cy) × Z / f
    Z = f × baseline / d
"""
print(pointcloud_info)

# Create approximate Q matrix for reprojection
# (In real applications, use cv2.stereoRectify to get proper Q)
h, w = left_gray.shape
focal_length = w  # Approximate
baseline = 1.0    # Normalized baseline
cx, cy = w / 2, h / 2

Q = np.float32([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0, focal_length],
    [0, 0, -1/baseline, 0]
])

# Reproject to 3D
points_3d = cv2.reprojectImageTo3D(disparity, Q)

# Filter valid points
mask = (disparity > 0) & (disparity < disparity.max() * 0.9)
valid_points = points_3d[mask]
valid_colors = left_img[mask] if len(left_img.shape) == 3 else np.stack([left_img[mask]]*3, axis=-1)

print(f"  Total pixels: {h * w}")
print(f"  Valid 3D points: {len(valid_points)}")


# =============================================================================
# 5. FEATURE-BASED SPARSE RECONSTRUCTION
# =============================================================================
print("\n--- 5. Feature-Based Sparse Reconstruction ---")

sparse_info = """
Sparse Reconstruction Pipeline:

1. Detect features (SIFT, ORB, etc.)
2. Match features between views
3. Triangulate matched points

Advantages:
  - Faster than dense reconstruction
  - Good for large-scale scenes
  - Useful for camera pose estimation

For full SFM pipeline, see 05_sfm_concepts.py
"""
print(sparse_info)

# Detect features
try:
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(left_gray, None)
    kp2, desc2 = sift.detectAndCompute(right_gray, None)
    detector_name = "SIFT"
except:
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, desc1 = orb.detectAndCompute(left_gray, None)
    kp2, desc2 = orb.detectAndCompute(right_gray, None)
    detector_name = "ORB"

print(f"  Detector: {detector_name}")
print(f"  Features in left image: {len(kp1)}")
print(f"  Features in right image: {len(kp2)}")

# Match features
if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
    if detector_name == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = bf.match(desc1, desc2)

    print(f"  Good matches: {len(good_matches)}")

    # Get matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Simple triangulation using disparity
    sparse_3d = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        d = x1 - x2  # Disparity
        if d > 0:
            Z = focal_length * baseline / d
            X = (x1 - cx) * Z / focal_length
            Y = (y1 - cy) * Z / focal_length
            sparse_3d.append([X, Y, Z])

    sparse_3d = np.array(sparse_3d) if sparse_3d else np.array([]).reshape(0, 3)
    print(f"  Sparse 3D points: {len(sparse_3d)}")
else:
    good_matches = []
    pts1, pts2 = np.array([]), np.array([])
    sparse_3d = np.array([]).reshape(0, 3)


# =============================================================================
# 6. SAVE POINT CLOUD AS PLY
# =============================================================================
print("\n--- 6. Saving Point Cloud ---")


def write_ply(filename, points, colors):
    """Write point cloud to PLY file."""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for pt, col in zip(points, colors):
            # Limit to reasonable depth range
            if abs(pt[2]) < 10000:
                r, g, b = int(col[2]), int(col[1]), int(col[0])  # BGR to RGB
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {r} {g} {b}\n")

    return filename


# Subsample for reasonable file size
max_points = 50000
if len(valid_points) > max_points:
    indices = np.random.choice(len(valid_points), max_points, replace=False)
    save_points = valid_points[indices]
    save_colors = valid_colors[indices]
else:
    save_points = valid_points
    save_colors = valid_colors

ply_file = os.path.join(os.path.dirname(__file__), f"pointcloud_{dataset_name.lower()}.ply")
write_ply(ply_file, save_points, save_colors)
print(f"  Saved {len(save_points)} points to: {ply_file}")
print("  Open in MeshLab, CloudCompare, or Blender to view")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display 3D reconstruction demonstrations."""

    # Normalize disparity for display
    disp_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_display, cv2.COLORMAP_JET)

    # Create stereo pair display
    if len(left_img.shape) == 3:
        left_display = left_img.copy()
        right_display = right_img.copy()
    else:
        left_display = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        right_display = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

    # Add labels
    cv2.putText(left_display, f"Left Image ({dataset_name})", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(right_display, "Right Image", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(disp_color, "Disparity Map (Red=near, Blue=far)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show stereo pair
    stereo_display = np.hstack([left_display, right_display])
    cv2.imshow("Stereo Pair", stereo_display)

    # Show disparity
    cv2.imshow("Disparity Map", disp_color)

    # Show feature matches if available
    if len(good_matches) > 0:
        match_img = cv2.drawMatches(
            left_display, kp1, right_display, kp2,
            good_matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.putText(match_img, f"Feature Matches ({len(good_matches)} total)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Feature Matches (Sparse Reconstruction)", match_img)

    # Create depth visualization (top-down view of point cloud)
    if len(valid_points) > 0:
        top_view = np.zeros((400, 500, 3), dtype=np.uint8)

        # Normalize points for display
        x_pts = valid_points[:, 0]
        z_pts = valid_points[:, 2]

        # Filter extreme values
        z_valid = (z_pts > np.percentile(z_pts, 5)) & (z_pts < np.percentile(z_pts, 95))
        x_pts = x_pts[z_valid]
        z_pts = z_pts[z_valid]
        colors_sub = valid_colors[z_valid]

        if len(x_pts) > 0:
            # Subsample for visualization
            step = max(1, len(x_pts) // 5000)
            x_pts = x_pts[::step]
            z_pts = z_pts[::step]
            colors_sub = colors_sub[::step]

            # Scale to image
            x_min, x_max = x_pts.min(), x_pts.max()
            z_min, z_max = z_pts.min(), z_pts.max()

            for x, z, col in zip(x_pts, z_pts, colors_sub):
                px = int((x - x_min) / (x_max - x_min + 1e-6) * 480 + 10)
                py = int((z - z_min) / (z_max - z_min + 1e-6) * 380 + 10)
                if 0 <= px < 500 and 0 <= py < 400:
                    cv2.circle(top_view, (px, 399 - py), 1,
                              (int(col[0]), int(col[1]), int(col[2])), -1)

        cv2.putText(top_view, "3D Point Cloud (Top-Down View)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(top_view, "X-axis horizontal, Z-axis vertical (depth)", (10, 385),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.imshow("3D Reconstruction (Top View)", top_view)

    print("\n3D Reconstruction Demo:")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Disparity range: [{disparity.min():.1f}, {disparity.max():.1f}]")
    print(f"  - 3D points generated: {len(valid_points)}")
    print(f"  - Point cloud saved: {ply_file}")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running 3D reconstruction with real stereo images...")
    print("=" * 60)
    show_demo()
