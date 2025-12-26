"""
Module 11: Blending Techniques for Panorama Stitching
=====================================================
Comparing different blending methods to eliminate visible seams.

This module demonstrates:
1. No Blending (hard edge) - Shows the seam problem
2. Alpha Blending - Linear gradient transition
3. Feather Blending - Distance-weighted combination
4. Multi-band Blending - Best quality using Laplacian pyramids
5. Side-by-side comparison of all methods

Topics Covered:
- Why blending is necessary
- Linear vs distance-weighted blending
- Image pyramids (Gaussian and Laplacian)
- OpenCV's built-in blending classes
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 11: Blending Techniques")
print("=" * 60)


def load_and_prepare_images():
    """Load images and compute homography for stitching demo."""
    # Try Blender Suzanne images (good overlap for blending demo)
    img1 = get_image("Blender_Suzanne1.jpg")
    img2 = get_image("Blender_Suzanne2.jpg")

    if img1 is None or img2 is None:
        # Try boat images
        img1 = get_image("boat1.jpg")
        img2 = get_image("boat2.jpg")

    if img1 is None or img2 is None:
        # Create synthetic images with obvious color difference
        print("Creating synthetic images for blending demo...")
        print("Run: python curriculum/sample_data/download_samples.py")

        # Two overlapping images with different brightness
        base = np.zeros((300, 500, 3), dtype=np.uint8)

        # Image 1: Slightly blue tint, left side
        img1 = base[:, :300].copy()
        img1[:, :] = (100, 80, 60)  # Bluish
        cv2.circle(img1, (100, 150), 50, (255, 200, 150), -1)
        cv2.rectangle(img1, (180, 100), (280, 200), (200, 255, 200), -1)

        # Image 2: Slightly warm tint, right side
        img2 = base[:, 100:400].copy()
        img2[:, :] = (60, 80, 100)  # Warmer
        cv2.circle(img2, (100, 150), 50, (150, 200, 255), -1)
        cv2.rectangle(img2, (180, 100), (280, 200), (200, 200, 255), -1)

        return img1, img2, None, 100  # No homography needed, 100px overlap

    # Compute homography for real images
    print(f"Using images: {img1.shape}, {img2.shape}")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"Found {len(good)} good matches")

    if len(good) < 4:
        return img1, img2, None, 100

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return img1, img2, H, None


# Load images
img1, img2, H, fixed_overlap = load_and_prepare_images()


# =============================================================================
# 1. NO BLENDING (HARD EDGE)
# =============================================================================
print("\n" + "=" * 60)
print("1. No Blending (Hard Edge)")
print("=" * 60)

no_blend_explanation = """
Simple Overlay:
  Just copy pixels from one image on top of another.

  Problems:
  - Visible seam at image boundary
  - Brightness/color discontinuity obvious
  - Any misalignment becomes very visible

  When acceptable:
  - Quick preview during development
  - Images with identical exposure
  - No visible overlap region
"""
print(no_blend_explanation)


def simple_overlay_stitch(img1, img2, H=None, overlap=100):
    """Stitch with no blending - just overlay."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if H is not None:
        # Use homography
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners1_t = cv2.perspectiveTransform(corners1, H)

        all_corners = np.vstack([corners1_t.reshape(-1, 2),
                                  [[0, 0], [0, h2], [w2, h2], [w2, 0]]])
        min_x, min_y = all_corners.min(axis=0)
        max_x, max_y = all_corners.max(axis=0)

        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0

        T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
        H_adj = T @ H

        out_w = int(max_x - min_x)
        out_h = int(max_y - min_y)

        result = cv2.warpPerspective(img1, H_adj, (out_w, out_h))
        x, y = int(offset_x), int(offset_y)
        result[y:y+h2, x:x+w2] = img2
    else:
        # Simple horizontal stitch with overlap
        total_w = w1 + w2 - overlap
        result = np.zeros((max(h1, h2), total_w, 3), dtype=np.uint8)
        result[:h1, :w1] = img1
        result[:h2, w1-overlap:w1-overlap+w2] = img2  # Overwrites overlap

    return result


result_no_blend = simple_overlay_stitch(img1, img2, H, fixed_overlap or 100)
print(f"No-blend result: {result_no_blend.shape}")


# =============================================================================
# 2. ALPHA BLENDING
# =============================================================================
print("\n" + "=" * 60)
print("2. Alpha Blending (Linear Gradient)")
print("=" * 60)

alpha_explanation = """
Alpha Blending:
  In the overlap region, linearly interpolate between images.

  result = (1 - alpha) * img1 + alpha * img2

  Where alpha goes from 0 to 1 across the overlap region.

  Pros:
  - Simple to implement
  - Smooth transition
  - Works well for similar exposures

  Cons:
  - Ghosting if images not perfectly aligned
  - Can blur high-frequency details
  - Linear gradient may be visible on gradients
"""
print(alpha_explanation)


def alpha_blend_stitch(img1, img2, H=None, overlap=100):
    """Stitch with linear alpha blending in overlap region."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if H is not None:
        # Warp first image
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners1_t = cv2.perspectiveTransform(corners1, H)

        all_corners = np.vstack([corners1_t.reshape(-1, 2),
                                  [[0, 0], [0, h2], [w2, h2], [w2, 0]]])
        min_x, min_y = all_corners.min(axis=0)
        max_x, max_y = all_corners.max(axis=0)

        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0

        T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
        H_adj = T @ H

        out_w = int(max_x - min_x)
        out_h = int(max_y - min_y)

        warped1 = cv2.warpPerspective(img1, H_adj, (out_w, out_h))
        mask1 = cv2.warpPerspective(np.ones_like(img1[:, :, 0]) * 255, H_adj, (out_w, out_h))

        # Place img2
        warped2 = np.zeros_like(warped1)
        mask2 = np.zeros((out_h, out_w), dtype=np.uint8)
        x, y = int(offset_x), int(offset_y)
        warped2[y:y+h2, x:x+w2] = img2
        mask2[y:y+h2, x:x+w2] = 255

        # Create alpha masks with gradient in overlap
        # Find overlap region
        overlap_mask = cv2.bitwise_and(mask1, mask2)

        # Create distance transforms for blending weights
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)

        # Normalize to create alpha
        total = dist1 + dist2 + 1e-6
        alpha1 = dist1 / total
        alpha2 = dist2 / total

        # Blend
        result = (warped1 * alpha1[:, :, np.newaxis] +
                  warped2 * alpha2[:, :, np.newaxis]).astype(np.uint8)

    else:
        # Simple horizontal with linear gradient
        h = max(h1, h2)
        total_w = w1 + w2 - overlap
        result = np.zeros((h, total_w, 3), dtype=np.uint8)

        # Non-overlapping regions
        result[:h1, :w1-overlap] = img1[:, :w1-overlap]
        result[:h2, w1:] = img2[:, overlap:]

        # Overlapping region with linear alpha
        for i in range(overlap):
            alpha = i / overlap
            x_out = w1 - overlap + i
            x1 = w1 - overlap + i
            x2 = i
            result[:min(h1, h2), x_out] = (
                (1 - alpha) * img1[:min(h1, h2), x1] +
                alpha * img2[:min(h1, h2), x2]
            ).astype(np.uint8)

    return result


result_alpha = alpha_blend_stitch(img1, img2, H, fixed_overlap or 100)
print(f"Alpha blend result: {result_alpha.shape}")


# =============================================================================
# 3. FEATHER BLENDING
# =============================================================================
print("\n" + "=" * 60)
print("3. Feather Blending (Distance-Weighted)")
print("=" * 60)

feather_explanation = """
Feather Blending:
  Weight each pixel by its distance from the image edge.
  Pixels near the center of an image have more weight.

  weight = distance_from_edge

  Benefits:
  - Natural falloff at boundaries
  - Center of each image preserved
  - Handles irregular overlap shapes

  OpenCV: cv2.detail.FeatherBlender()
"""
print(feather_explanation)


def feather_blend_stitch(img1, img2, H=None, overlap=100, feather_amount=100):
    """Stitch with feather (distance-weighted) blending."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if H is not None:
        # Similar setup as alpha blend
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners1_t = cv2.perspectiveTransform(corners1, H)

        all_corners = np.vstack([corners1_t.reshape(-1, 2),
                                  [[0, 0], [0, h2], [w2, h2], [w2, 0]]])
        min_x, min_y = all_corners.min(axis=0)
        max_x, max_y = all_corners.max(axis=0)

        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0

        T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
        H_adj = T @ H

        out_w = int(max_x - min_x)
        out_h = int(max_y - min_y)

        warped1 = cv2.warpPerspective(img1, H_adj, (out_w, out_h))
        mask1 = cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8) * 255, H_adj, (out_w, out_h))

        warped2 = np.zeros_like(warped1)
        mask2 = np.zeros((out_h, out_w), dtype=np.uint8)
        x, y = int(offset_x), int(offset_y)
        warped2[y:y+h2, x:x+w2] = img2
        mask2[y:y+h2, x:x+w2] = 255

        # Distance transform for feathering
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)

        # Clip distances for feather effect
        dist1 = np.clip(dist1, 0, feather_amount)
        dist2 = np.clip(dist2, 0, feather_amount)

        total = dist1 + dist2 + 1e-6
        w1_map = dist1 / total
        w2_map = dist2 / total

        result = (warped1 * w1_map[:, :, np.newaxis] +
                  warped2 * w2_map[:, :, np.newaxis]).astype(np.uint8)
    else:
        # Simplified horizontal feather
        h = max(h1, h2)
        total_w = w1 + w2 - overlap
        result = np.zeros((h, total_w, 3), dtype=np.uint8)

        # Create feather weights
        weight1 = np.zeros((h, total_w), dtype=np.float32)
        weight2 = np.zeros((h, total_w), dtype=np.float32)

        # Weight 1: full on left, fades in overlap
        weight1[:h1, :w1-overlap] = 1.0
        for i in range(overlap):
            weight1[:h1, w1-overlap+i] = 1.0 - (i / overlap)

        # Weight 2: fades in overlap, full on right
        for i in range(overlap):
            weight2[:h2, w1-overlap+i] = i / overlap
        weight2[:h2, w1:] = 1.0

        # Normalize
        total = weight1 + weight2 + 1e-6
        weight1 /= total
        weight2 /= total

        # Expand images to canvas
        canvas1 = np.zeros((h, total_w, 3), dtype=np.float32)
        canvas2 = np.zeros((h, total_w, 3), dtype=np.float32)
        canvas1[:h1, :w1] = img1
        canvas2[:h2, w1-overlap:w1-overlap+w2] = img2

        result = (canvas1 * weight1[:, :, np.newaxis] +
                  canvas2 * weight2[:, :, np.newaxis]).astype(np.uint8)

    return result


result_feather = feather_blend_stitch(img1, img2, H, fixed_overlap or 100)
print(f"Feather blend result: {result_feather.shape}")


# =============================================================================
# 4. MULTI-BAND BLENDING
# =============================================================================
print("\n" + "=" * 60)
print("4. Multi-band Blending (Laplacian Pyramid)")
print("=" * 60)

multiband_explanation = """
Multi-band Blending:
  The best quality blending, used by professional stitching software.

  Key Insight:
  - Low frequencies (smooth gradients): Blend broadly
  - High frequencies (edges, details): Blend narrowly

  How it works:
  1. Build Gaussian pyramid for each image (progressively blur + downsample)
  2. Build Laplacian pyramid (difference between levels = high-freq details)
  3. Build Gaussian pyramid for blend mask
  4. Blend each pyramid level with corresponding mask level
  5. Reconstruct from blended Laplacian pyramid

  Why it works:
  - Smooth brightness transitions (low freq blended broadly)
  - Sharp edges preserved (high freq blended at boundary only)
  - No ghosting, no blurring

  OpenCV: cv2.detail.MultiBandBlender()
"""
print(multiband_explanation)


def build_gaussian_pyramid(img, levels):
    """Build Gaussian pyramid."""
    pyramid = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def build_laplacian_pyramid(img, levels):
    """Build Laplacian pyramid."""
    gaussian = build_gaussian_pyramid(img, levels)
    laplacian = []

    for i in range(levels):
        # Laplacian = Gaussian - Upsampled(next level)
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        expanded = cv2.pyrUp(gaussian[i + 1], dstsize=size)
        lap = cv2.subtract(gaussian[i], expanded)
        laplacian.append(lap)

    # Top of pyramid (smallest)
    laplacian.append(gaussian[-1])
    return laplacian


def reconstruct_from_laplacian(pyramid):
    """Reconstruct image from Laplacian pyramid."""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = cv2.add(img, pyramid[i])
    return img


def multiband_blend_simple(img1, img2, mask, levels=4):
    """Multi-band blending for two images with a mask."""
    # Build Laplacian pyramids for both images
    lap1 = build_laplacian_pyramid(img1.astype(np.float32), levels)
    lap2 = build_laplacian_pyramid(img2.astype(np.float32), levels)

    # Build Gaussian pyramid for the mask
    mask_pyr = build_gaussian_pyramid(mask.astype(np.float32) / 255.0, levels)

    # Blend each level
    blended_lap = []
    for l1, l2, m in zip(lap1, lap2, mask_pyr):
        # Ensure mask has 3 channels if images are color
        if len(l1.shape) == 3 and len(m.shape) == 2:
            m = m[:, :, np.newaxis]
        blended = l1 * (1 - m) + l2 * m
        blended_lap.append(blended)

    # Reconstruct
    result = reconstruct_from_laplacian(blended_lap)
    return np.clip(result, 0, 255).astype(np.uint8)


def multiband_blend_stitch(img1, img2, H=None, overlap=100, levels=4):
    """Stitch with multi-band blending."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if H is not None:
        # Warp setup
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners1_t = cv2.perspectiveTransform(corners1, H)

        all_corners = np.vstack([corners1_t.reshape(-1, 2),
                                  [[0, 0], [0, h2], [w2, h2], [w2, 0]]])
        min_x, min_y = all_corners.min(axis=0)
        max_x, max_y = all_corners.max(axis=0)

        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0

        T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
        H_adj = T @ H

        out_w = int(max_x - min_x)
        out_h = int(max_y - min_y)

        warped1 = cv2.warpPerspective(img1, H_adj, (out_w, out_h))
        mask1 = cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8) * 255, H_adj, (out_w, out_h))

        warped2 = np.zeros_like(warped1)
        mask2 = np.zeros((out_h, out_w), dtype=np.uint8)
        x, y = int(offset_x), int(offset_y)
        warped2[y:y+h2, x:x+w2] = img2
        mask2[y:y+h2, x:x+w2] = 255

        # Create blend mask (gradient in overlap)
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
        blend_mask = (dist2 / (dist1 + dist2 + 1e-6) * 255).astype(np.uint8)

        result = multiband_blend_simple(warped1, warped2, blend_mask, levels)
    else:
        # Simple horizontal case
        h = max(h1, h2)
        total_w = w1 + w2 - overlap

        # Expand images to same canvas
        canvas1 = np.zeros((h, total_w, 3), dtype=np.uint8)
        canvas2 = np.zeros((h, total_w, 3), dtype=np.uint8)
        canvas1[:h1, :w1] = img1
        canvas2[:h2, w1-overlap:w1-overlap+w2] = img2

        # Create gradient mask
        mask = np.zeros((h, total_w), dtype=np.uint8)
        mask[:, w1-overlap//2:] = 255  # Right side is img2

        result = multiband_blend_simple(canvas1, canvas2, mask, levels)

    return result


result_multiband = multiband_blend_stitch(img1, img2, H, fixed_overlap or 100)
print(f"Multi-band blend result: {result_multiband.shape}")


# =============================================================================
# 5. COMPARISON SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("5. Blending Methods Comparison")
print("=" * 60)

comparison_table = """
| Method          | Quality | Speed  | Use Case                    |
|-----------------|---------|--------|------------------------------|
| No Blending     | Poor    | Fast   | Quick preview, testing       |
| Alpha Blending  | Fair    | Fast   | Similar exposure images      |
| Feather Blend   | Good    | Medium | General purpose              |
| Multi-band      | Best    | Slow   | Professional quality needed  |

Recommendations:
- Real-time: Alpha blending
- Quality: Multi-band blending
- Balanced: Feather blending

OpenCV Stitcher uses Multi-band by default.
"""
print(comparison_table)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display all blending methods for comparison."""
    # Resize all results to same height for comparison
    target_h = 250

    def resize_to_height(img, h):
        aspect = img.shape[1] / img.shape[0]
        return cv2.resize(img, (int(h * aspect), h))

    # Input images
    input_display = np.hstack([
        cv2.resize(img1, (200, 150)),
        cv2.resize(img2, (200, 150))
    ])
    cv2.putText(input_display, "Image 1", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(input_display, "Image 2", (210, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("Input Images", input_display)

    # Individual results
    r1 = resize_to_height(result_no_blend, target_h)
    cv2.putText(r1, "No Blending (visible seam)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("1. No Blending", r1)

    r2 = resize_to_height(result_alpha, target_h)
    cv2.putText(r2, "Alpha Blending", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("2. Alpha Blending", r2)

    r3 = resize_to_height(result_feather, target_h)
    cv2.putText(r3, "Feather Blending", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("3. Feather Blending", r3)

    r4 = resize_to_height(result_multiband, target_h)
    cv2.putText(r4, "Multi-band Blending (Best)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("4. Multi-band Blending", r4)

    # Side by side comparison (2x2 grid)
    grid_h = 180
    g1 = resize_to_height(result_no_blend, grid_h)
    g2 = resize_to_height(result_alpha, grid_h)
    g3 = resize_to_height(result_feather, grid_h)
    g4 = resize_to_height(result_multiband, grid_h)

    # Make all same width
    max_w = max(g1.shape[1], g2.shape[1], g3.shape[1], g4.shape[1])
    def pad_to_width(img, w):
        if img.shape[1] < w:
            pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
            return np.hstack([img, pad])
        return img[:, :w]

    g1 = pad_to_width(g1, max_w)
    g2 = pad_to_width(g2, max_w)
    g3 = pad_to_width(g3, max_w)
    g4 = pad_to_width(g4, max_w)

    # Add labels
    cv2.putText(g1, "No Blend", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(g2, "Alpha", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(g3, "Feather", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(g4, "Multi-band", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    row1 = np.hstack([g1, g2])
    row2 = np.hstack([g3, g4])
    comparison = np.vstack([row1, row2])
    cv2.imshow("Comparison (2x2)", comparison)

    print("\n" + "=" * 60)
    print("Blending Techniques Demonstration Complete!")
    print("=" * 60)
    print("\nWindows:")
    print("  1. No Blending - Visible hard edge/seam")
    print("  2. Alpha Blending - Linear gradient transition")
    print("  3. Feather Blending - Distance-weighted blend")
    print("  4. Multi-band Blending - Best quality (Laplacian pyramid)")
    print("  Comparison - All methods side by side")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_demo()
