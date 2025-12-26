"""
Module 11: Manual Image Stitching Pipeline
==========================================
Understanding each step of the stitching process in detail.

This module breaks down the stitching pipeline into discrete steps:
1. Feature Detection (SIFT vs ORB comparison)
2. Feature Matching (BFMatcher vs FLANN, ratio test)
3. Homography Estimation (RANSAC visualization)
4. Image Warping (coordinate transforms)
5. Simple Overlay (seam problem demonstration)

Topics Covered:
- Feature detector comparison for stitching
- Match filtering with Lowe's ratio test
- RANSAC for robust homography estimation
- Understanding the homography matrix
- Calculating output canvas size
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 11: Manual Stitching Pipeline")
print("=" * 60)


def load_image_pair():
    """Load a pair of overlapping images for stitching."""
    # Try box images (good for homography demo)
    box = get_image("box.png")
    box_scene = get_image("box_in_scene.png")
    if box is not None and box_scene is not None:
        print("Using: box.png and box_in_scene.png")
        return box, box_scene, "box"

    # Try boat images (panorama sequence)
    boat1 = get_image("boat1.jpg")
    boat2 = get_image("boat2.jpg")
    if boat1 is not None and boat2 is not None:
        print("Using: boat1.jpg and boat2.jpg")
        return boat1, boat2, "boat"

    # Try Blender Suzanne images
    suz1 = get_image("Blender_Suzanne1.jpg")
    suz2 = get_image("Blender_Suzanne2.jpg")
    if suz1 is not None and suz2 is not None:
        print("Using: Blender_Suzanne1.jpg and Blender_Suzanne2.jpg")
        return suz1, suz2, "suzanne"

    # Fallback: Create synthetic images
    print("No sample images found. Creating synthetic overlapping images.")
    print("Run: python curriculum/sample_data/download_samples.py")

    # Create scene with distinctive features
    scene = np.zeros((400, 800, 3), dtype=np.uint8)
    for i in range(800):
        scene[:, i] = (40 + i // 10, 60, 120 - i // 10)

    # Add features for matching
    cv2.circle(scene, (150, 200), 60, (0, 200, 255), -1)
    cv2.rectangle(scene, (250, 150), (350, 280), (255, 100, 0), -1)
    cv2.circle(scene, (450, 200), 70, (0, 255, 100), -1)
    cv2.rectangle(scene, (550, 120), (680, 300), (100, 0, 255), -1)

    # Add texture pattern
    for y in range(0, 400, 40):
        for x in range(0, 800, 40):
            cv2.circle(scene, (x, y), 3, (200, 200, 200), -1)

    # Create overlapping views
    img1 = scene[:, 0:450].copy()
    img2 = scene[:, 300:750].copy()

    return img1, img2, "synthetic"


# Load images
img1, img2, image_type = load_image_pair()
print(f"Image 1: {img1.shape}")
print(f"Image 2: {img2.shape}")


# =============================================================================
# 1. FEATURE DETECTION COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("1. Feature Detection: SIFT vs ORB")
print("=" * 60)

# SIFT (Scale-Invariant Feature Transform)
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(img1, None)
kp2_sift, des2_sift = sift.detectAndCompute(img2, None)

print(f"\nSIFT Features:")
print(f"  Image 1: {len(kp1_sift)} keypoints")
print(f"  Image 2: {len(kp2_sift)} keypoints")
print(f"  Descriptor shape: {des1_sift.shape if des1_sift is not None else 'None'}")
print(f"  Descriptor type: float32 (128-dimensional)")

# ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create(nfeatures=2000)
kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)

print(f"\nORB Features:")
print(f"  Image 1: {len(kp1_orb)} keypoints")
print(f"  Image 2: {len(kp2_orb)} keypoints")
print(f"  Descriptor shape: {des1_orb.shape if des1_orb is not None else 'None'}")
print(f"  Descriptor type: binary (32 bytes = 256 bits)")

comparison = """
SIFT vs ORB for Stitching:

| Aspect          | SIFT                | ORB                 |
|-----------------|---------------------|---------------------|
| Speed           | Slower              | Much faster         |
| Accuracy        | Higher              | Good                |
| Scale Invariant | Yes                 | Limited             |
| Rotation        | Full 360            | Full 360            |
| Descriptor      | 128 float (512B)    | 32 bytes binary     |
| Matching        | L2 distance         | Hamming distance    |
| License         | Free (OpenCV 4.4+)  | Always free         |

Recommendation:
- SIFT: When accuracy matters (panoramas, 3D reconstruction)
- ORB: Real-time applications, mobile devices
"""
print(comparison)

# Draw keypoints comparison
kp_img1_sift = cv2.drawKeypoints(img1, kp1_sift[:100], None,
                                  color=(0, 255, 0),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_img1_orb = cv2.drawKeypoints(img1, kp1_orb[:100], None,
                                 color=(0, 255, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# =============================================================================
# 2. FEATURE MATCHING: BFMatcher vs FLANN
# =============================================================================
print("\n" + "=" * 60)
print("2. Feature Matching")
print("=" * 60)

# Use SIFT for better matching quality
kp1, des1 = kp1_sift, des1_sift
kp2, des2 = kp2_sift, des2_sift

# BFMatcher (Brute Force)
print("\n--- BFMatcher (Brute Force) ---")
bf = cv2.BFMatcher(cv2.NORM_L2)
bf_matches = bf.knnMatch(des1, des2, k=2)
print(f"Total matches: {len(bf_matches)}")

# FLANN (Fast Library for Approximate Nearest Neighbors)
print("\n--- FLANN Matcher ---")
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Higher = more accurate but slower

flann = cv2.FlannBasedMatcher(index_params, search_params)
flann_matches = flann.knnMatch(des1, des2, k=2)
print(f"Total matches: {len(flann_matches)}")

matcher_comparison = """
BFMatcher vs FLANN:

| Aspect          | BFMatcher           | FLANN               |
|-----------------|---------------------|---------------------|
| Algorithm       | Exhaustive search   | Approximate NN      |
| Speed (large)   | Slow O(n*m)         | Fast O(log n)       |
| Accuracy        | 100% exact          | ~99% approximate    |
| Best for        | Small datasets      | Large datasets      |
| Memory          | Low                 | Higher (tree index) |

For stitching:
- Few images: BFMatcher is fine
- Many images/features: FLANN is faster
"""
print(matcher_comparison)


# =============================================================================
# 3. RATIO TEST (Lowe's Test)
# =============================================================================
print("\n" + "=" * 60)
print("3. Lowe's Ratio Test")
print("=" * 60)

ratio_explanation = """
Lowe's Ratio Test:
  For each keypoint, we find the 2 nearest matches (knnMatch k=2).
  If the best match is significantly better than the second-best,
  it's likely a correct match.

  Ratio = distance(best) / distance(second_best)
  Accept if ratio < threshold (typically 0.7-0.8)

  Why it works:
  - Correct matches: best match is much closer than alternatives
  - Incorrect matches: multiple similar-looking features exist
"""
print(ratio_explanation)

# Apply ratio test with different thresholds
for ratio_thresh in [0.6, 0.7, 0.8, 0.9]:
    good_matches = []
    for m, n in bf_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    print(f"Ratio threshold {ratio_thresh}: {len(good_matches)} good matches")

# Use 0.75 for final matching
good_matches = []
for m, n in bf_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"\nUsing ratio=0.75: {len(good_matches)} good matches retained")


# =============================================================================
# 4. HOMOGRAPHY ESTIMATION WITH RANSAC
# =============================================================================
print("\n" + "=" * 60)
print("4. Homography Estimation (RANSAC)")
print("=" * 60)

if len(good_matches) >= 4:
    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Count inliers vs outliers
    inliers = mask.ravel().sum()
    outliers = len(mask) - inliers
    print(f"\nRANSAC Results:")
    print(f"  Total matches: {len(good_matches)}")
    print(f"  Inliers: {inliers} ({100*inliers/len(good_matches):.1f}%)")
    print(f"  Outliers: {outliers} ({100*outliers/len(good_matches):.1f}%)")

    ransac_explanation = """
RANSAC (Random Sample Consensus):
  1. Randomly select 4 point pairs (minimum for homography)
  2. Compute homography from these 4 pairs
  3. Count how many other points agree (inliers)
  4. Repeat many times, keep best homography
  5. Re-estimate using all inliers

  Parameters:
  - cv2.RANSAC: Use RANSAC method
  - 5.0: Maximum reprojection error (pixels) for a point to be inlier

  Alternatives:
  - cv2.LMEDS: Least-Median robust method
  - cv2.RHO: PROSAC-like faster variant
    """
    print(ransac_explanation)

    # Explain the homography matrix
    print("\nHomography Matrix H:")
    print(H)

    h_explanation = """
Homography Matrix Elements:
  H = | h00  h01  h02 |
      | h10  h11  h12 |
      | h20  h21  h22 |

  - h00, h01, h10, h11: Rotation and scale
  - h02, h12: Translation (x, y shift)
  - h20, h21: Perspective distortion
  - h22: Usually 1 (normalization)

  Transformation: [x', y', w'] = H * [x, y, 1]
  Final coords: (x'/w', y'/w')
    """
    print(h_explanation)

    # Separate inliers and outliers for visualization
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    outlier_matches = [good_matches[i] for i in range(len(good_matches)) if not mask[i]]

else:
    print(f"Not enough matches ({len(good_matches)}) for homography estimation!")
    H = None
    inlier_matches = []
    outlier_matches = []


# =============================================================================
# 5. IMAGE WARPING (Step by Step)
# =============================================================================
print("\n" + "=" * 60)
print("5. Image Warping")
print("=" * 60)

if H is not None:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Step 1: Transform corner points of img1
    print("\nStep 1: Transform corners of Image 1")
    corners1 = np.float32([
        [0, 0],           # Top-left
        [0, h1],          # Bottom-left
        [w1, h1],         # Bottom-right
        [w1, 0]           # Top-right
    ]).reshape(-1, 1, 2)

    corners1_transformed = cv2.perspectiveTransform(corners1, H)
    print(f"  Original corners:\n{corners1.reshape(-1, 2)}")
    print(f"  Transformed corners:\n{corners1_transformed.reshape(-1, 2)}")

    # Step 2: Calculate canvas bounds
    print("\nStep 2: Calculate output canvas size")
    all_corners = np.vstack([
        corners1_transformed.reshape(-1, 2),
        np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]])
    ])

    min_x, min_y = all_corners.min(axis=0)
    max_x, max_y = all_corners.max(axis=0)

    print(f"  X range: {min_x:.1f} to {max_x:.1f}")
    print(f"  Y range: {min_y:.1f} to {max_y:.1f}")

    # Step 3: Create translation matrix for negative coordinates
    print("\nStep 3: Handle negative coordinates")
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    print(f"  Translation offset: ({offset_x:.1f}, {offset_y:.1f})")

    # Translation matrix
    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    # Combined transform: first H, then translate
    H_adjusted = T @ H

    # Output size
    out_width = int(max_x - min_x)
    out_height = int(max_y - min_y)
    print(f"  Output size: {out_width} x {out_height}")

    # Step 4: Warp image 1
    print("\nStep 4: Warp Image 1")
    warped1 = cv2.warpPerspective(img1, H_adjusted, (out_width, out_height))
    print(f"  Warped shape: {warped1.shape}")

    # Step 5: Place image 2
    print("\nStep 5: Place Image 2 (simple overlay)")
    result_overlay = warped1.copy()
    x_start = int(offset_x)
    y_start = int(offset_y)
    result_overlay[y_start:y_start+h2, x_start:x_start+w2] = img2

    print(f"  Final result: {result_overlay.shape}")

else:
    print("Skipping warping (no valid homography)")
    warped1 = img1.copy()
    result_overlay = np.hstack([img1, img2])


# =============================================================================
# 6. SEAM PROBLEM DEMONSTRATION
# =============================================================================
print("\n" + "=" * 60)
print("6. The Seam Problem")
print("=" * 60)

seam_explanation = """
Why Simple Overlay Creates Visible Seams:

1. Brightness Differences:
   - Different exposure between images
   - Vignetting at image edges

2. Color Inconsistency:
   - White balance variations
   - Different lighting conditions

3. Geometric Misalignment:
   - Imperfect homography estimation
   - Lens distortion not corrected

4. Hard Boundary:
   - Abrupt transition at image border
   - Eye easily detects discontinuity

Solutions (covered in 03_blending_techniques.py):
- Alpha blending: Gradual transition
- Feather blending: Distance-weighted
- Multi-band blending: Frequency-based
- Seam finding: Optimal cut path
"""
print(seam_explanation)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display all stages of manual stitching."""

    # 1. Input images
    input_display = np.hstack([
        cv2.resize(img1, (300, 225)),
        cv2.resize(img2, (300, 225))
    ])
    cv2.putText(input_display, "Image 1", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(input_display, "Image 2", (310, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("1. Input Images", input_display)

    # 2. Feature detection comparison
    feat_display = np.hstack([
        cv2.resize(kp_img1_sift, (300, 225)),
        cv2.resize(kp_img1_orb, (300, 225))
    ])
    cv2.putText(feat_display, "SIFT", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(feat_display, "ORB", (310, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("2. Feature Detection", feat_display)

    # 3. Feature matches (all good matches)
    if len(good_matches) > 0:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None,
                                     matchColor=(0, 255, 0),
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        match_display = cv2.resize(match_img, (800, 300))
        cv2.putText(match_display, f"Good Matches: {len(good_matches)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("3. Feature Matches", match_display)

    # 4. RANSAC inliers vs outliers
    if len(inlier_matches) > 0:
        # Draw inliers in green, outliers in red
        inlier_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:30], None,
                                      matchColor=(0, 255, 0),
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if len(outlier_matches) > 0:
            outlier_img = cv2.drawMatches(img1, kp1, img2, kp2, outlier_matches[:20], inlier_img,
                                           matchColor=(0, 0, 255),
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else:
            outlier_img = inlier_img

        ransac_display = cv2.resize(outlier_img, (800, 300))
        cv2.putText(ransac_display, f"RANSAC: {len(inlier_matches)} inliers (green)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(ransac_display, f"{len(outlier_matches)} outliers (red)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("4. RANSAC Inliers/Outliers", ransac_display)

    # 5. Warped image
    if H is not None:
        warp_display = cv2.resize(warped1, (600, 300))
        cv2.putText(warp_display, "Warped Image 1", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("5. Warped Image", warp_display)

    # 6. Final overlay (with visible seam)
    final_h = min(400, result_overlay.shape[0])
    final_w = int(final_h * result_overlay.shape[1] / result_overlay.shape[0])
    final_display = cv2.resize(result_overlay, (final_w, final_h))
    cv2.putText(final_display, "Simple Overlay (notice seam)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("6. Result (Seam Visible)", final_display)

    print("\n" + "=" * 60)
    print("Manual Stitching Pipeline Complete!")
    print("=" * 60)
    print("\nWindow Summary:")
    print("  1. Input Images - The two source images")
    print("  2. Feature Detection - SIFT (green) vs ORB (yellow)")
    print("  3. Feature Matches - Good matches after ratio test")
    print("  4. RANSAC - Inliers (green) vs outliers (red)")
    print("  5. Warped Image - Image 1 transformed by homography")
    print("  6. Result - Simple overlay showing visible seam")
    print("\nNext: See 03_blending_techniques.py for seam removal")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_demo()
