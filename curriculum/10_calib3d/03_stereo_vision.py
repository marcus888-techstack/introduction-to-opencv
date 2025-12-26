"""
Module 7: Camera Calibration - Stereo Vision
=============================================
Stereo camera calibration, rectification, and depth estimation.

Official Docs: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html

Topics Covered:
1. Stereo Camera Concepts
2. Stereo Calibration
3. Stereo Rectification
4. Disparity Maps
5. Depth Estimation
"""

import cv2
import numpy as np
import os
import sys
import urllib.request

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 7: Stereo Vision")
print("=" * 60)


# =============================================================================
# 1. STEREO VISION CONCEPTS
# =============================================================================
print("\n--- 1. Stereo Vision Concepts ---")

stereo_concepts = """
Stereo Vision Overview:

Two cameras capture the same scene from different viewpoints.
The displacement (disparity) between corresponding points reveals depth.

Key Concepts:
  - Baseline (B): Distance between camera centers
  - Disparity (d): Pixel difference between left and right images
  - Depth (Z): Distance to object

Depth Formula:
  Z = (f × B) / d

  Where:
    Z = depth (distance to object)
    f = focal length (in pixels)
    B = baseline (camera separation)
    d = disparity (pixel difference)

Larger disparity → Closer object
Smaller disparity → Farther object

Pipeline:
  1. Calibrate each camera individually
  2. Stereo calibrate (find relative pose)
  3. Stereo rectify (align images horizontally)
  4. Compute disparity map
  5. Convert disparity to depth
"""
print(stereo_concepts)


# =============================================================================
# 2. LOAD STEREO IMAGES (Middlebury Benchmark)
# =============================================================================
print("\n--- 2. Loading Stereo Images (Middlebury Benchmark) ---")

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sample_data')

# Middlebury Stereo Benchmark Datasets
# https://vision.middlebury.edu/stereo/data/
MIDDLEBURY_DATASETS = {
    "tsukuba": {
        "left": "https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/scene1.row3.col1.ppm",
        "right": "https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/scene1.row3.col3.ppm",
        "disp": "https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/truedisp.row3.col3.pgm",
        "disp_scale": 16,  # Disparity scale factor
        "description": "Classic 384x288 indoor scene"
    },
    "cones": {
        "left": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png",
        "right": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png",
        "disp": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/disp2.png",
        "disp_scale": 4,  # Quarter-pixel accuracy, scale by 4
        "description": "450x375 colorful cones scene"
    },
    "teddy": {
        "left": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy/im2.png",
        "right": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy/im6.png",
        "disp": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy/disp2.png",
        "disp_scale": 4,
        "description": "450x375 teddy bear scene"
    }
}


def download_middlebury_dataset(dataset_name="tsukuba"):
    """Download Middlebury stereo benchmark dataset."""
    if dataset_name not in MIDDLEBURY_DATASETS:
        print(f"  Unknown dataset: {dataset_name}")
        return None, None, None

    dataset = MIDDLEBURY_DATASETS[dataset_name]
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    files = {}
    for key in ["left", "right", "disp"]:
        url = dataset[key]
        ext = url.split(".")[-1]
        filename = f"{dataset_name}_{key}.{ext}"
        filepath = os.path.join(SAMPLE_DIR, filename)

        if not os.path.exists(filepath):
            try:
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"  Failed to download {filename}: {e}")
                filepath = None
        files[key] = filepath

    return files.get("left"), files.get("right"), files.get("disp")


def load_stereo_pair(dataset_name="tsukuba"):
    """Load a Middlebury stereo benchmark pair."""
    left_path, right_path, disp_path = download_middlebury_dataset(dataset_name)

    if left_path and right_path and os.path.exists(left_path) and os.path.exists(right_path):
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)
        disp_gt = cv2.imread(disp_path, cv2.IMREAD_GRAYSCALE) if disp_path else None

        dataset = MIDDLEBURY_DATASETS[dataset_name]
        print(f"  Loaded Middlebury '{dataset_name}' dataset")
        print(f"  Description: {dataset['description']}")
        return left, right, disp_gt, dataset.get("disp_scale", 1)

    # Fallback: create synthetic stereo pair
    print("  Creating synthetic stereo pair...")
    left = np.zeros((300, 400, 3), dtype=np.uint8)
    right = np.zeros((300, 400, 3), dtype=np.uint8)

    # Add objects at different depths (different disparities)
    cv2.rectangle(left, (100, 100), (150, 200), (0, 0, 255), -1)
    cv2.rectangle(right, (70, 100), (120, 200), (0, 0, 255), -1)  # 30px disparity
    cv2.rectangle(left, (250, 100), (300, 200), (0, 255, 0), -1)
    cv2.rectangle(right, (240, 100), (290, 200), (0, 255, 0), -1)  # 10px disparity

    return left, right, None, 1


# Load stereo pair (try different datasets)
DATASET = "tsukuba"  # Options: "tsukuba", "cones", "teddy"
left_img, right_img, ground_truth_disp, disp_scale = load_stereo_pair(DATASET)
if left_img is not None:
    print(f"  Image size: {left_img.shape}")


# =============================================================================
# 3. STEREO CALIBRATION CONCEPTS
# =============================================================================
print("\n--- 3. Stereo Calibration ---")

stereo_calib_info = """
Stereo Calibration Process:

1. Calibrate Left Camera:
   ret1, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(...)

2. Calibrate Right Camera:
   ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(...)

3. Stereo Calibrate (find R, T between cameras):
   ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
       objpoints,      # 3D points (same for both)
       imgpoints_left,  # 2D points in left images
       imgpoints_right, # 2D points in right images
       K1, dist1,       # Left camera intrinsics
       K2, dist2,       # Right camera intrinsics
       imageSize,
       flags=cv2.CALIB_FIX_INTRINSIC  # Keep intrinsics fixed
   )

Output:
  R - Rotation matrix (3x3) from left to right camera
  T - Translation vector (3x1) from left to right camera
  E - Essential matrix
  F - Fundamental matrix

Baseline = ||T|| (magnitude of translation)
"""
print(stereo_calib_info)


# =============================================================================
# 4. STEREO RECTIFICATION
# =============================================================================
print("\n--- 4. Stereo Rectification ---")

rectification_info = """
Stereo Rectification:

Purpose: Transform images so that:
  - Epipolar lines become horizontal
  - Corresponding points are on the same row
  - Simplifies matching (1D search instead of 2D)

Rectification Visualization:
  Before:               After:
  ┌─────────────┐      ┌─────────────┐
  │    /        │      │ ────────────│
  │   /  epipolar      │ ────────────│ horizontal
  │  /   lines  │  →   │ ────────────│ epipolar
  │ /           │      │ ────────────│ lines
  └─────────────┘      └─────────────┘

cv2.stereoRectify():
  R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
      K1, dist1, K2, dist2,
      imageSize, R, T,
      alpha=0  # 0=only valid pixels, 1=all pixels
  )

Output:
  R1, R2 - Rotation matrices for each camera
  P1, P2 - Projection matrices (3x4)
  Q      - Disparity-to-depth mapping matrix (4x4)
  roi1, roi2 - Valid pixel regions

Create rectification maps:
  map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, size, cv2.CV_32FC1)
  map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, size, cv2.CV_32FC1)

Apply:
  left_rectified = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
  right_rectified = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
"""
print(rectification_info)


# =============================================================================
# 5. DISPARITY MAP COMPUTATION
# =============================================================================
print("\n--- 5. Disparity Map ---")

# Convert to grayscale for disparity computation
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img

disparity_info = """
Disparity Computation Methods:

1. Block Matching (BM) - Fast, less accurate:
   stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
   disparity = stereo.compute(left, right)

2. Semi-Global Block Matching (SGBM) - Better quality:
   stereo = cv2.StereoSGBM_create(
       minDisparity=0,
       numDisparities=64,     # Must be divisible by 16
       blockSize=5,
       P1=8 * 3 * blockSize**2,   # Penalty for small disparity changes
       P2=32 * 3 * blockSize**2,  # Penalty for large disparity changes
       disp12MaxDiff=1,
       uniquenessRatio=10,
       speckleWindowSize=100,
       speckleRange=32
   )
   disparity = stereo.compute(left, right)

Note: disparity is in fixed-point format, divide by 16 for actual values
"""
print(disparity_info)

# Create stereo matchers
print("\nComputing disparity maps...")

# Method 1: Block Matching (fast)
stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity_bm = stereo_bm.compute(left_gray, right_gray)

# Method 2: Semi-Global Block Matching (better quality)
block_size = 5
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=block_size,
    P1=8 * 3 * block_size**2,
    P2=32 * 3 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity_sgbm = stereo_sgbm.compute(left_gray, right_gray)

print(f"  BM disparity range: [{disparity_bm.min()}, {disparity_bm.max()}]")
print(f"  SGBM disparity range: [{disparity_sgbm.min()}, {disparity_sgbm.max()}]")


# =============================================================================
# 6. DISPARITY TO DEPTH
# =============================================================================
print("\n--- 6. Disparity to Depth Conversion ---")

depth_info = """
Converting Disparity to Depth:

Method 1: Using Q matrix (from stereoRectify):
  depth_3d = cv2.reprojectImageTo3D(disparity, Q)
  # depth_3d[:,:,2] contains Z (depth) values

Method 2: Direct formula:
  Z = (f × B) / d

  Where:
    f = focal length (pixels) - from camera matrix K[0,0]
    B = baseline (meters) - magnitude of T from stereoCalibrate
    d = disparity (pixels)

Example:
  focal_length = 700  # pixels
  baseline = 0.1      # 10 cm between cameras

  # Convert disparity (fixed-point) to float
  disparity_float = disparity.astype(np.float32) / 16.0

  # Avoid division by zero
  disparity_float[disparity_float <= 0] = 0.1

  # Calculate depth
  depth = (focal_length * baseline) / disparity_float
"""
print(depth_info)

# Convert disparity to depth (example with assumed parameters)
focal_length = 700  # pixels (typical for these sample images)
baseline = 0.1      # 10 cm (assumed)

# Convert to float disparity
disparity_float = disparity_sgbm.astype(np.float32) / 16.0
disparity_float[disparity_float <= 0] = 0.1

# Calculate depth
depth_map = (focal_length * baseline) / disparity_float
depth_map = np.clip(depth_map, 0, 10)  # Clip to reasonable range

print(f"  Depth range: [{depth_map.min():.2f}m, {depth_map.max():.2f}m]")


# =============================================================================
# 7. WLS FILTER (WEIGHTED LEAST SQUARES)
# =============================================================================
print("\n--- 7. Disparity Filtering ---")

wls_info = """
WLS Filter for Better Disparity:

The WLS (Weighted Least Squares) filter improves disparity maps:
- Fills holes
- Smooths while preserving edges
- Uses both left and right disparity

Usage:
  # Create matchers
  left_matcher = cv2.StereoSGBM_create(...)
  right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

  # Compute disparities
  left_disp = left_matcher.compute(left, right)
  right_disp = right_matcher.compute(right, left)

  # Create and apply WLS filter
  wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
  wls_filter.setLambda(8000)
  wls_filter.setSigmaColor(1.5)
  filtered_disp = wls_filter.filter(left_disp, left, disparity_map_right=right_disp)

Note: Requires opencv-contrib-python (cv2.ximgproc)
"""
print(wls_info)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display stereo vision demonstrations."""

    # Normalize disparity for display
    disp_bm_display = cv2.normalize(disparity_bm, None, 0, 255, cv2.NORM_MINMAX)
    disp_bm_display = np.uint8(disp_bm_display)

    disp_sgbm_display = cv2.normalize(disparity_sgbm, None, 0, 255, cv2.NORM_MINMAX)
    disp_sgbm_display = np.uint8(disp_sgbm_display)

    # Apply colormap for better visualization
    disp_bm_color = cv2.applyColorMap(disp_bm_display, cv2.COLORMAP_JET)
    disp_sgbm_color = cv2.applyColorMap(disp_sgbm_display, cv2.COLORMAP_JET)

    # Normalize depth for display
    depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = np.uint8(depth_display)
    depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

    # Resize for display
    h, w = left_img.shape[:2]
    display_size = (400, 300)

    left_resized = cv2.resize(left_img, display_size)
    right_resized = cv2.resize(right_img, display_size)
    disp_bm_resized = cv2.resize(disp_bm_color, display_size)
    disp_sgbm_resized = cv2.resize(disp_sgbm_color, display_size)
    depth_resized = cv2.resize(depth_color, display_size)

    # Create stereo pair comparison
    stereo_pair = np.hstack([left_resized, right_resized])
    cv2.putText(stereo_pair, f"Left ({DATASET})", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(stereo_pair, f"Right ({DATASET})", (410, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Middlebury Stereo Pair", stereo_pair)

    # Ground truth comparison (if available)
    if ground_truth_disp is not None:
        gt_display = cv2.normalize(ground_truth_disp, None, 0, 255, cv2.NORM_MINMAX)
        gt_display = np.uint8(gt_display)
        gt_color = cv2.applyColorMap(gt_display, cv2.COLORMAP_JET)
        gt_resized = cv2.resize(gt_color, display_size)

        # Compare SGBM with ground truth
        comparison = np.hstack([disp_sgbm_resized, gt_resized])
        cv2.putText(comparison, "SGBM (Computed)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "Ground Truth", (410, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Disparity: Computed vs Ground Truth", comparison)

    # Disparity comparison (BM vs SGBM)
    disparity_comparison = np.hstack([disp_bm_resized, disp_sgbm_resized])
    cv2.putText(disparity_comparison, "Block Matching", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(disparity_comparison, "SGBM", (410, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Disparity: BM vs SGBM", disparity_comparison)

    # Depth map
    cv2.putText(depth_resized, "Depth Map", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Depth Map (Blue=far, Red=close)", depth_resized)

    # Draw horizontal lines to show epipolar geometry
    stereo_lines = stereo_pair.copy()
    for y in range(0, stereo_lines.shape[0], 30):
        cv2.line(stereo_lines, (0, y), (stereo_lines.shape[1], y), (0, 255, 0), 1)
    cv2.putText(stereo_lines, "Epipolar lines (matching points on same row)", (10, stereo_lines.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow("Epipolar Lines", stereo_lines)

    print(f"\nDataset: Middlebury '{DATASET}'")
    print(f"  Source: https://vision.middlebury.edu/stereo/data/")
    print("\nDisparity Map Legend:")
    print("  Blue = Far objects (small disparity)")
    print("  Red = Close objects (large disparity)")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running stereo vision demonstrations...")
    print("=" * 60)
    show_demo()
