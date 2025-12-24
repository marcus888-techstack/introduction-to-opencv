"""
Module 11: Image Stitching - Panorama Creation
===============================================
Creating panoramic images from multiple photos.

Official Docs: https://docs.opencv.org/4.x/d1/d46/group__stitching.html

Topics Covered:
1. High-Level Stitcher API
2. Manual Stitching Pipeline
3. Image Alignment
4. Blending
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 11: Image Stitching")
print("=" * 60)


def create_test_images():
    """Create overlapping images for stitching demo."""
    # Create a wide scene
    full_scene = np.zeros((300, 800, 3), dtype=np.uint8)

    # Background gradient
    for i in range(800):
        full_scene[:, i] = (50 + i // 8, 80, 150 - i // 10)

    # Add distinctive features
    cv2.circle(full_scene, (100, 150), 50, (0, 255, 255), -1)
    cv2.rectangle(full_scene, (200, 80), (300, 180), (255, 0, 0), -1)
    cv2.circle(full_scene, (400, 150), 60, (0, 255, 0), -1)
    cv2.rectangle(full_scene, (500, 100), (600, 200), (0, 0, 255), -1)
    cv2.circle(full_scene, (700, 150), 40, (255, 255, 0), -1)

    # Add text landmarks
    cv2.putText(full_scene, "LEFT", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(full_scene, "CENTER", (350, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(full_scene, "RIGHT", (650, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Split into overlapping images
    img1 = full_scene[:, 0:350].copy()    # Left
    img2 = full_scene[:, 250:550].copy()  # Center
    img3 = full_scene[:, 450:800].copy()  # Right

    return [img1, img2, img3], full_scene


images, full_scene = create_test_images()
print(f"Created {len(images)} overlapping images")
for i, img in enumerate(images):
    print(f"  Image {i+1}: {img.shape}")


# =============================================================================
# 1. HIGH-LEVEL STITCHER API
# =============================================================================
print("\n--- 1. High-Level Stitcher API ---")

# Create stitcher
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# Alternative modes:
# cv2.Stitcher_PANORAMA - For panoramic images
# cv2.Stitcher_SCANS    - For flat document scans

# Stitch images
status, panorama = stitcher.stitch(images)

status_codes = {
    cv2.Stitcher_OK: "OK - Stitching successful",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR - Need more images",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR - Homography estimation failed",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR - Camera params adjustment failed"
}

print(f"Stitching status: {status_codes.get(status, 'Unknown')}")

if status == cv2.Stitcher_OK:
    print(f"Panorama size: {panorama.shape}")
else:
    # Fallback: create dummy panorama
    panorama = np.hstack(images)


# =============================================================================
# 2. MANUAL STITCHING PIPELINE
# =============================================================================
print("\n--- 2. Manual Stitching Pipeline ---")

pipeline_info = """
Manual Stitching Steps:

1. Feature Detection:
   detector = cv2.SIFT_create() or cv2.ORB_create()
   keypoints, descriptors = detector.detectAndCompute(img, None)

2. Feature Matching:
   matcher = cv2.BFMatcher()
   matches = matcher.knnMatch(desc1, desc2, k=2)
   # Apply ratio test

3. Homography Estimation:
   H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

4. Image Warping:
   warped = cv2.warpPerspective(img, H, output_size)

5. Blending:
   - Simple: Average overlapping regions
   - Alpha blending: Weighted combination
   - Multi-band blending: Best quality
"""
print(pipeline_info)


def manual_stitch(img1, img2):
    """Manually stitch two images."""
    # Detect features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Good matches: {len(good)}")

    if len(good) < 4:
        print("Not enough matches for homography")
        return np.hstack([img1, img2])

    # Get point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp first image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate output size
    pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    # Determine canvas size
    min_x = min(0, dst[:, 0, 0].min())
    min_y = min(0, dst[:, 0, 1].min())
    max_x = max(w2, dst[:, 0, 0].max())
    max_y = max(h2, dst[:, 0, 1].max())

    # Translation matrix
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # Warp
    output_size = (int(max_x - min_x), int(max_y - min_y))
    warped = cv2.warpPerspective(img1, translation @ H, output_size)

    # Place second image
    x_offset = int(-min_x)
    y_offset = int(-min_y)
    warped[y_offset:y_offset+h2, x_offset:x_offset+w2] = img2

    return warped


# Try manual stitching
manual_result = manual_stitch(images[0], images[1])
print(f"Manual stitch result: {manual_result.shape}")


# =============================================================================
# 3. STITCHER CONFIGURATION
# =============================================================================
print("\n--- 3. Stitcher Configuration ---")

config_info = """
Stitcher Configuration:

# Create with mode
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# Feature detection
stitcher.setFeaturesFinder(cv2.detail.OrbFeaturesFinder())
# or SIFT: cv2.detail.SiftFeaturesFinder()

# Feature matching
# stitcher.setFeaturesMatcher(matcher)

# Warping
stitcher.setWarper(cv2.PyRotationWarper('spherical', 1.0))
# Types: spherical, cylindrical, plane, fisheye

# Seam estimation
# stitcher.setSeamEstimator(estimator)

# Blending
stitcher.setBlender(cv2.detail.MultiBandBlender())
# or: cv2.detail.FeatherBlender()

# Composition resolution
stitcher.setCompositingResol(-1)  # -1 for original
stitcher.setRegistrationResol(0.6)  # Resolution for matching
"""
print(config_info)


# =============================================================================
# 4. BLENDING TECHNIQUES
# =============================================================================
print("\n--- 4. Blending Techniques ---")

blending_info = """
Blending Methods:

1. No Blending (Simple Copy):
   - Just overlay images
   - Visible seams

2. Alpha Blending:
   - Linear interpolation in overlap
   - Better but gradient visible

3. Feather Blending:
   - Distance-based weighting
   - Smooth transitions

4. Multi-band Blending:
   - Blend different frequencies separately
   - Best quality, most complex
   - Use cv2.detail.MultiBandBlender()

5. Seam Cutting:
   - Find optimal seam in overlap
   - Minimize visible transitions
"""
print(blending_info)


# Simple alpha blending demo
def alpha_blend(img1, img2, overlap_width):
    """Simple alpha blending demonstration."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Assume images have same height
    h = min(h1, h2)
    total_width = w1 + w2 - overlap_width

    result = np.zeros((h, total_width, 3), dtype=np.uint8)

    # Copy non-overlapping parts
    result[:h, :w1-overlap_width] = img1[:h, :w1-overlap_width]
    result[:h, w1:] = img2[:h, overlap_width:]

    # Blend overlapping region
    for i in range(overlap_width):
        alpha = i / overlap_width
        x1 = w1 - overlap_width + i
        x2 = i
        result[:h, w1-overlap_width+i] = (
            (1 - alpha) * img1[:h, x1] + alpha * img2[:h, x2]
        ).astype(np.uint8)

    return result


blended = alpha_blend(images[0], images[1], 100)
print(f"Alpha blended result: {blended.shape}")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display stitching demos."""

    # Show input images
    input_display = np.hstack([
        cv2.resize(img, (200, 150)) for img in images
    ])
    cv2.putText(input_display, "Image 1", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(input_display, "Image 2", (210, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(input_display, "Image 3", (410, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("Input Images", input_display)

    # Show panorama result
    if status == cv2.Stitcher_OK:
        pano_display = cv2.resize(panorama, (600, 200))
        cv2.putText(pano_display, "Stitched Panorama", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Panorama", pano_display)

    # Show original full scene
    full_display = cv2.resize(full_scene, (600, 200))
    cv2.putText(full_display, "Original Full Scene", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Original Scene", full_display)

    # Show alpha blending result
    blend_display = cv2.resize(blended, (400, 150))
    cv2.putText(blend_display, "Alpha Blended", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("Alpha Blending", blend_display)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running image stitching demonstrations...")
    print("=" * 60)
    show_demo()
