"""
Module 11: Cylindrical and Spherical Panoramas
===============================================
Creating wide-angle and 360-degree panoramas using different projections.

This module covers:
1. Why planar projection fails for wide panoramas
2. Cylindrical projection for 360-degree horizontal panoramas
3. Spherical projection for full spherical coverage
4. Using OpenCV warpers for projection
5. Multi-image panorama stitching with projections

Topics Covered:
- Projection types and when to use each
- Cylindrical coordinate transformation
- Spherical mapping for VR/360 content
- OpenCV's PyRotationWarper API
"""

import cv2
import numpy as np
import os
import sys
import math

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 11: Cylindrical and Spherical Panoramas")
print("=" * 60)


# =============================================================================
# 1. PROJECTION TYPES EXPLAINED
# =============================================================================
print("\n" + "=" * 60)
print("1. Projection Types")
print("=" * 60)

projection_explanation = """
Three Main Projection Types for Panoramas:

1. PLANAR (Perspective) Projection:
   - Standard perspective like a flat photo
   - Works for: Small angle (<90 degree) panoramas
   - Problems: Extreme distortion at edges for wide views
   - Lines stay straight

2. CYLINDRICAL Projection:
   - Project onto a cylinder wrapped around camera
   - Works for: 360-degree horizontal panoramas
   - Horizontal lines curve near top/bottom
   - Vertical lines stay straight
   - Used by: Google Street View (horizontal)

3. SPHERICAL (Equirectangular) Projection:
   - Project onto a sphere, unwrap to rectangle
   - Works for: Full 360 x 180 degree coverage
   - Both horizontal and vertical lines curve (except at center)
   - Used by: VR headsets, 360-degree video
   - Aspect ratio: 2:1 (360 wide x 180 tall)

When to use each:
- Small rotation (<60 deg): Planar is fine
- Wide horizontal (>90 deg): Cylindrical
- Full environment: Spherical
"""
print(projection_explanation)


def load_multiple_images():
    """Load multiple images for panorama demo."""
    # Try boat sequence
    images = []
    for i in range(1, 7):
        img = get_image(f"boat{i}.jpg")
        if img is not None:
            images.append(img)

    if len(images) >= 3:
        print(f"Loaded {len(images)} boat images")
        return images, "boat"

    # Try with a single wide image split into parts
    building = get_image("building.jpg")
    if building is not None:
        h, w = building.shape[:2]
        # Split into 4 overlapping images
        step = w // 5
        overlap = step // 2
        parts = []
        for i in range(4):
            start = i * step
            end = min(start + step + overlap, w)
            parts.append(building[:, start:end].copy())
        print(f"Split building.jpg into {len(parts)} parts")
        return parts, "building"

    # Synthetic test images
    print("Creating synthetic panorama images...")
    print("Run: python curriculum/sample_data/download_samples.py")

    # Create a wide scene and split into parts
    scene = np.zeros((300, 1200, 3), dtype=np.uint8)

    # Background gradient simulating sky
    for y in range(300):
        blue = max(0, 200 - y)
        scene[y, :] = (blue, 100, 50 + y // 3)

    # Add some "landmarks" for stitching
    colors = [(0, 200, 255), (255, 100, 0), (0, 255, 100), (200, 0, 200)]
    for i, (x, color) in enumerate(zip([150, 400, 700, 1000], colors)):
        cv2.circle(scene, (x, 150), 50, color, -1)
        cv2.putText(scene, str(i+1), (x-15, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Split into overlapping parts
    parts = []
    for i in range(4):
        start = i * 250
        end = start + 400
        parts.append(scene[:, start:end].copy())

    return parts, "synthetic"


# Load images
images, image_source = load_multiple_images()
print(f"Working with {len(images)} images from: {image_source}")
for i, img in enumerate(images):
    print(f"  Image {i+1}: {img.shape}")


# =============================================================================
# 2. CYLINDRICAL PROJECTION
# =============================================================================
print("\n" + "=" * 60)
print("2. Cylindrical Projection")
print("=" * 60)

cylindrical_math = """
Cylindrical Projection Math:

Given a point (x, y) in the image with principal point (cx, cy)
and focal length f:

1. Convert to normalized camera coordinates:
   x_norm = (x - cx) / f
   y_norm = (y - cy) / f

2. Project onto cylinder:
   theta = atan(x_norm)           # Angle around cylinder
   h = y_norm / sqrt(1 + x_norm^2)  # Height on cylinder

3. Convert back to image coordinates:
   x_cyl = f * theta + cx
   y_cyl = f * h + cy

Effect:
- Horizontal FOV can extend to 360 degrees
- Vertical lines remain vertical (important for architecture)
- Horizontal lines curve (barrel distortion effect)
"""
print(cylindrical_math)


def cylindrical_warp(img, focal_length=None):
    """
    Warp image to cylindrical projection.

    Args:
        img: Input image
        focal_length: Camera focal length in pixels.
                     If None, estimate as image_width * 0.5
    """
    h, w = img.shape[:2]

    if focal_length is None:
        # Estimate focal length (typically 0.5-1.5 times image width)
        focal_length = w * 0.7

    # Principal point (center of image)
    cx, cy = w / 2, h / 2

    # Create output image (can be wider for 360 panoramas)
    out_w = w
    out_h = h
    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Create mapping arrays
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    for y_out in range(out_h):
        for x_out in range(out_w):
            # Convert output coords to cylindrical
            theta = (x_out - cx) / focal_length
            h_cyl = (y_out - cy) / focal_length

            # Convert back to input (planar) coordinates
            x_in = focal_length * np.tan(theta) + cx
            y_in = focal_length * h_cyl / np.cos(theta) + cy

            map_x[y_out, x_out] = x_in
            map_y[y_out, x_out] = y_in

    # Apply remapping
    result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)

    return result


# Warp first image to cylindrical
img_cylindrical = cylindrical_warp(images[0])
print(f"Cylindrical warped: {img_cylindrical.shape}")


# =============================================================================
# 3. SPHERICAL PROJECTION
# =============================================================================
print("\n" + "=" * 60)
print("3. Spherical Projection")
print("=" * 60)

spherical_math = """
Spherical (Equirectangular) Projection:

Projects the image onto a sphere centered at the camera.

1. Convert to normalized camera coordinates:
   x_norm = (x - cx) / f
   y_norm = (y - cy) / f

2. Convert to spherical coordinates:
   theta = atan(x_norm)                    # Azimuth (-pi to pi)
   phi = atan(y_norm / sqrt(1 + x_norm^2)) # Elevation (-pi/2 to pi/2)

3. Map to equirectangular output:
   x_out = (theta / pi + 1) * width / 2
   y_out = (phi / (pi/2) + 1) * height / 2

Output format:
- Full sphere: width = 2 * height (360 x 180 degrees)
- Used for VR and 360 video
"""
print(spherical_math)


def spherical_warp(img, focal_length=None):
    """
    Warp image to spherical projection.
    """
    h, w = img.shape[:2]

    if focal_length is None:
        focal_length = w * 0.7

    cx, cy = w / 2, h / 2

    result = np.zeros((h, w, 3), dtype=np.uint8)
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y_out in range(h):
        for x_out in range(w):
            # Output coords to spherical
            theta = (x_out - cx) / focal_length
            phi = (y_out - cy) / focal_length

            # Spherical to planar
            x_in = focal_length * np.tan(theta) + cx
            y_in = focal_length * np.tan(phi) / np.cos(theta) + cy

            map_x[y_out, x_out] = x_in
            map_y[y_out, x_out] = y_in

    result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)

    return result


# Warp first image to spherical
img_spherical = spherical_warp(images[0])
print(f"Spherical warped: {img_spherical.shape}")


# =============================================================================
# 4. OPENCV WARPERS
# =============================================================================
print("\n" + "=" * 60)
print("4. OpenCV PyRotationWarper")
print("=" * 60)

warper_explanation = """
OpenCV provides built-in warpers through cv2.PyRotationWarper:

Available warper types:
- 'plane'       - Planar (perspective) projection
- 'cylindrical' - Cylindrical projection
- 'spherical'   - Spherical (equirectangular) projection
- 'fisheye'     - Fisheye projection
- 'stereographic' - Stereographic projection
- 'compressedPlaneA2B1' - Compressed planar variations
- 'transverseMercator' - Map-style projection

Usage:
    warper = cv2.PyRotationWarper('cylindrical', focal_length)
    corner, warped = warper.warp(image, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

Parameters:
- K: Camera intrinsic matrix (3x3)
- R: Rotation matrix (3x3) - orientation of this image
- Returns: corner position and warped image
"""
print(warper_explanation)


def create_camera_matrix(img, focal_length=None):
    """Create a simple camera intrinsic matrix."""
    h, w = img.shape[:2]
    if focal_length is None:
        focal_length = w * 0.7

    K = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    return K, focal_length


def demo_opencv_warpers():
    """Demonstrate different OpenCV warper types."""
    img = images[0]
    K, f = create_camera_matrix(img)

    # Identity rotation (camera looking straight ahead)
    R = np.eye(3, dtype=np.float32)

    warper_types = ['plane', 'cylindrical', 'spherical']
    results = {}

    print("\nApplying different warpers:")
    for wtype in warper_types:
        try:
            warper = cv2.PyRotationWarper(wtype, f)
            corner, warped = warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            results[wtype] = warped
            print(f"  {wtype}: {warped.shape}")
        except Exception as e:
            print(f"  {wtype}: Failed ({e})")
            results[wtype] = img.copy()

    return results


warped_results = demo_opencv_warpers()


# =============================================================================
# 5. MULTI-IMAGE CYLINDRICAL PANORAMA
# =============================================================================
print("\n" + "=" * 60)
print("5. Multi-Image Cylindrical Panorama")
print("=" * 60)


def cylindrical_stitch_multiple(images, focal_length=None):
    """
    Stitch multiple images using cylindrical projection.

    Process:
    1. Warp all images to cylindrical
    2. Find features and match between consecutive pairs
    3. Estimate horizontal translations
    4. Blend all images together
    """
    if len(images) < 2:
        return images[0] if images else None

    # Estimate focal length from first image
    h, w = images[0].shape[:2]
    if focal_length is None:
        focal_length = w * 0.8

    # Warp all images to cylindrical
    print("Warping images to cylindrical projection...")
    warped_images = []
    for i, img in enumerate(images):
        warped = cylindrical_warp(img, focal_length)
        warped_images.append(warped)
        print(f"  Image {i+1} warped")

    # Find horizontal translations between consecutive images
    print("Finding translations between images...")
    sift = cv2.SIFT_create()
    translations = [0]  # First image at x=0

    for i in range(len(warped_images) - 1):
        img1 = warped_images[i]
        img2 = warped_images[i + 1]

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            # Fallback: assume 70% overlap
            translations.append(translations[-1] + int(img1.shape[1] * 0.3))
            print(f"  {i+1}->{i+2}: Using default translation")
            continue

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) < 4:
            translations.append(translations[-1] + int(img1.shape[1] * 0.3))
            print(f"  {i+1}->{i+2}: Not enough matches, using default")
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # For cylindrical, we mainly need horizontal translation
        # Use homography but extract translation
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None:
            # Translation is primarily the x-shift
            dx = -H[0, 2]  # Negative because we're placing img2 relative to img1
            translations.append(translations[-1] + int(dx))
            print(f"  {i+1}->{i+2}: dx = {dx:.1f}px")
        else:
            translations.append(translations[-1] + int(img1.shape[1] * 0.3))
            print(f"  {i+1}->{i+2}: Homography failed, using default")

    # Calculate output dimensions
    min_x = min(translations)
    max_x = max(t + warped_images[i].shape[1] for i, t in enumerate(translations))

    out_h = max(img.shape[0] for img in warped_images)
    out_w = int(max_x - min_x)

    print(f"Output panorama: {out_w} x {out_h}")

    # Create output canvas
    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    weights = np.zeros((out_h, out_w), dtype=np.float32)

    # Place images with simple averaging in overlap
    x_offset = -min_x
    for i, (img, tx) in enumerate(zip(warped_images, translations)):
        x_start = int(tx + x_offset)
        x_end = x_start + img.shape[1]
        h_img = img.shape[0]

        # Create weight map (center=1, edges=0)
        w_img = img.shape[1]
        weight = np.ones((h_img, w_img), dtype=np.float32)
        fade = min(50, w_img // 4)
        for f in range(fade):
            weight[:, f] = f / fade
            weight[:, w_img - 1 - f] = f / fade

        # Mask for valid (non-black) pixels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid = (gray > 10).astype(np.float32)
        weight *= valid

        # Accumulate
        for c in range(3):
            result[:h_img, x_start:x_end, c] = (
                result[:h_img, x_start:x_end, c] +
                img[:, :, c] * weight
            ).astype(np.uint8)
        weights[:h_img, x_start:x_end] += weight

    # Normalize
    weights = np.maximum(weights, 1e-6)  # Avoid division by zero
    for c in range(3):
        result[:, :, c] = (result[:, :, c].astype(np.float32) / weights).astype(np.uint8)

    return result


# Create cylindrical panorama
print("\nCreating multi-image cylindrical panorama...")
cylindrical_pano = cylindrical_stitch_multiple(images)
print(f"Cylindrical panorama: {cylindrical_pano.shape}")


# =============================================================================
# 6. USING OPENCV STITCHER WITH CUSTOM WARPER
# =============================================================================
print("\n" + "=" * 60)
print("6. OpenCV Stitcher with Custom Warper")
print("=" * 60)

stitcher_warper_info = """
Using custom warper with OpenCV Stitcher:

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

    # Set custom warper
    # Note: This requires knowing the focal length
    # stitcher.setWarper(cv2.PyRotationWarper('cylindrical', focal_length))

    # Alternative: Let Stitcher auto-select
    status, pano = stitcher.stitch(images)

The Stitcher automatically:
1. Estimates focal length from feature matches
2. Chooses appropriate warper based on field of view
3. Handles blending and seam finding
"""
print(stitcher_warper_info)

# Use OpenCV Stitcher for comparison
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, opencv_pano = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    print(f"OpenCV Stitcher result: {opencv_pano.shape}")
else:
    print(f"OpenCV Stitcher failed with status: {status}")
    opencv_pano = np.hstack([cv2.resize(img, (200, 150)) for img in images])


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display projection comparisons and panorama results."""

    # 1. Input images
    input_row = np.hstack([cv2.resize(img, (150, 100)) for img in images[:4]])
    cv2.putText(input_row, "Input Images", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("1. Input Images", input_row)

    # 2. Single image: Original vs Cylindrical vs Spherical
    orig = cv2.resize(images[0], (250, 180))
    cyl = cv2.resize(img_cylindrical, (250, 180))
    sph = cv2.resize(img_spherical, (250, 180))

    cv2.putText(orig, "Original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(cyl, "Cylindrical", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(sph, "Spherical", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    proj_compare = np.hstack([orig, cyl, sph])
    cv2.imshow("2. Projection Comparison", proj_compare)

    # 3. OpenCV Warpers
    if warped_results:
        warper_row = []
        for name in ['plane', 'cylindrical', 'spherical']:
            if name in warped_results:
                w = cv2.resize(warped_results[name], (220, 150))
                cv2.putText(w, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                warper_row.append(w)
        if warper_row:
            cv2.imshow("3. OpenCV PyRotationWarper", np.hstack(warper_row))

    # 4. Multi-image cylindrical panorama
    pano_h = 200
    pano_w = int(pano_h * cylindrical_pano.shape[1] / cylindrical_pano.shape[0])
    pano_display = cv2.resize(cylindrical_pano, (min(pano_w, 1000), pano_h))
    cv2.putText(pano_display, "Cylindrical Panorama (Manual)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("4. Cylindrical Panorama", pano_display)

    # 5. OpenCV Stitcher result
    if status == cv2.Stitcher_OK:
        ocv_h = 200
        ocv_w = int(ocv_h * opencv_pano.shape[1] / opencv_pano.shape[0])
        ocv_display = cv2.resize(opencv_pano, (min(ocv_w, 1000), ocv_h))
        cv2.putText(ocv_display, "OpenCV Stitcher (Auto)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("5. OpenCV Stitcher Result", ocv_display)

    print("\n" + "=" * 60)
    print("Cylindrical/Spherical Panorama Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  - Planar projection: Good for <90 degree FOV")
    print("  - Cylindrical: For wide horizontal panoramas (360 deg)")
    print("  - Spherical: For full environment capture (VR/360)")
    print("\nOpenCV Stitcher handles projection automatically,")
    print("but understanding projections helps troubleshoot issues.")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_demo()
