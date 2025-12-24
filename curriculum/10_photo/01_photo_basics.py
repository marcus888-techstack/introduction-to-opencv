"""
Module 10: Computational Photography - Basics
==============================================
Image enhancement and computational photography techniques.

Official Docs: https://docs.opencv.org/4.x/d0/d25/tutorial_table_of_content_photo.html

Topics Covered:
1. Inpainting (Image Restoration)
2. Denoising
3. HDR Imaging
4. Image Blending
5. Seamless Cloning
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 10: Computational Photography")
print("=" * 60)


def create_test_image():
    """Create test image with objects to demonstrate effects."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Gradient background
    for i in range(600):
        for j in range(400):
            img[j, i] = (100 + i // 6, 80 + j // 5, 150 - i // 8)

    # Add shapes
    cv2.circle(img, (200, 200), 80, (0, 200, 255), -1)
    cv2.rectangle(img, (350, 100), (500, 250), (255, 100, 50), -1)
    cv2.ellipse(img, (450, 320), (80, 50), 30, 0, 360, (100, 255, 100), -1)

    return img


original = create_test_image()


# =============================================================================
# 1. INPAINTING
# =============================================================================
print("\n--- 1. Inpainting ---")

# Create image with "damage" to inpaint
damaged = original.copy()
# Add some scratches
cv2.line(damaged, (100, 50), (500, 350), (0, 0, 0), 8)
cv2.line(damaged, (50, 200), (550, 250), (0, 0, 0), 8)
cv2.circle(damaged, (300, 200), 30, (0, 0, 0), -1)

# Create mask (white where we want to inpaint)
mask = np.zeros(damaged.shape[:2], dtype=np.uint8)
cv2.line(mask, (100, 50), (500, 350), 255, 10)
cv2.line(mask, (50, 200), (550, 250), 255, 10)
cv2.circle(mask, (300, 200), 32, 255, -1)

# Inpaint using Navier-Stokes method
inpainted_ns = cv2.inpaint(damaged, mask, inpaintRadius=5,
                            flags=cv2.INPAINT_NS)

# Inpaint using Alexandru Telea's method
inpainted_telea = cv2.inpaint(damaged, mask, inpaintRadius=5,
                               flags=cv2.INPAINT_TELEA)

print("Inpainting methods:")
print("  INPAINT_NS    - Navier-Stokes based")
print("  INPAINT_TELEA - Alexandru Telea's method (fast marching)")


# =============================================================================
# 2. DENOISING
# =============================================================================
print("\n--- 2. Denoising ---")

# Add noise to image
noisy = original.copy()
noise = np.random.normal(0, 30, noisy.shape).astype(np.int16)
noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Non-local means denoising (color image)
# Parameters: src, h, hForColorComponents, templateWindowSize, searchWindowSize
denoised = cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21)

# Grayscale denoising
gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
denoised_gray = cv2.fastNlMeansDenoising(gray_noisy, None, 10, 7, 21)

print("Denoising functions:")
print("  fastNlMeansDenoising()       - Grayscale")
print("  fastNlMeansDenoisingColored() - Color images")
print("  fastNlMeansDenoisingMulti()  - Video (multiple frames)")

denoise_params = """
Denoising Parameters:
  h                   - Filter strength (higher = more smoothing)
  hForColorComponents - Strength for color (usually same as h)
  templateWindowSize  - Size of template patch (7 is common)
  searchWindowSize    - Size of search area (21 is common)
"""
print(denoise_params)


# =============================================================================
# 3. HDR IMAGING
# =============================================================================
print("\n--- 3. HDR Imaging ---")

# Simulate multiple exposures
def create_exposures(base_img):
    """Create simulated multi-exposure images."""
    under = cv2.convertScaleAbs(base_img, alpha=0.5, beta=-30)
    normal = base_img.copy()
    over = cv2.convertScaleAbs(base_img, alpha=1.5, beta=50)
    return [under, normal, over]


exposures = create_exposures(original)
exposure_times = np.array([0.25, 1.0, 4.0], dtype=np.float32)

# Merge exposures using Debevec method
merge_debevec = cv2.createMergeDebevec()
# Note: This would need proper exposure times and calibration
# hdr = merge_debevec.process(exposures, times=exposure_times)

# Tonemap HDR to LDR for display
# tonemap = cv2.createTonemap(gamma=2.2)
# ldr = tonemap.process(hdr)

hdr_info = """
HDR Workflow:

1. Capture multiple exposures:
   - Under, normal, over exposed images
   - Record exposure times

2. Align images (if needed):
   alignMTB = cv2.createAlignMTB()
   alignMTB.process(images, images)

3. Merge to HDR:
   merge = cv2.createMergeDebevec()  # or MergeRobertson, MergeMertens
   hdr = merge.process(images, times)

4. Tonemap to LDR:
   tonemap = cv2.createTonemap(gamma=2.2)
   ldr = tonemap.process(hdr)

Alternative (Exposure Fusion - no HDR):
  mergeMertens = cv2.createMergeMertens()
  fusion = mergeMertens.process(images)
"""
print(hdr_info)

# Use Mertens for exposure fusion (doesn't need exposure times)
merge_mertens = cv2.createMergeMertens()
fusion = merge_mertens.process(exposures)
fusion_8bit = np.clip(fusion * 255, 0, 255).astype(np.uint8)


# =============================================================================
# 4. SEAMLESS CLONING
# =============================================================================
print("\n--- 4. Seamless Cloning ---")

# Create source image (object to clone)
source = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.circle(source, (50, 50), 40, (0, 255, 255), -1)
cv2.circle(source, (35, 35), 10, (50, 50, 50), -1)  # Eye
cv2.circle(source, (65, 35), 10, (50, 50, 50), -1)  # Eye
cv2.ellipse(source, (50, 65), (20, 10), 0, 0, 180, (50, 50, 50), 3)  # Smile

# Create mask
clone_mask = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(clone_mask, (50, 50), 45, 255, -1)

# Destination
dest = original.copy()

# Clone center point
center = (450, 300)

# Seamless clone - normal
try:
    cloned_normal = cv2.seamlessClone(source, dest, clone_mask, center,
                                       cv2.NORMAL_CLONE)
except cv2.error:
    cloned_normal = dest.copy()

# Seamless clone - mixed
try:
    cloned_mixed = cv2.seamlessClone(source, dest, clone_mask, center,
                                      cv2.MIXED_CLONE)
except cv2.error:
    cloned_mixed = dest.copy()

print("Seamless cloning modes:")
print("  NORMAL_CLONE    - Texture transfer")
print("  MIXED_CLONE     - Gradient mixing")
print("  MONOCHROME_TRANSFER - Texture transfer without color")


# =============================================================================
# 5. STYLIZATION
# =============================================================================
print("\n--- 5. Stylization ---")

# Edge-preserving filter
stylized = cv2.stylization(original, sigma_s=60, sigma_r=0.4)

# Pencil sketch
gray_pencil, color_pencil = cv2.pencilSketch(original, sigma_s=60, sigma_r=0.07,
                                              shade_factor=0.05)

# Detail enhancement
detailed = cv2.detailEnhance(original, sigma_s=10, sigma_r=0.15)

# Edge-preserving filtering
edge_preserved = cv2.edgePreservingFilter(original, flags=1, sigma_s=60, sigma_r=0.4)

print("Stylization functions:")
print("  stylization()        - Artistic effect")
print("  pencilSketch()       - Pencil drawing effect")
print("  detailEnhance()      - Enhance fine details")
print("  edgePreservingFilter() - Smooth while keeping edges")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display computational photography demos."""

    # Inpainting demo
    inpaint_display = np.hstack([
        cv2.resize(damaged, (300, 200)),
        cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (300, 200)),
        cv2.resize(inpainted_telea, (300, 200))
    ])
    labels = ["Damaged", "Mask", "Inpainted"]
    for i, label in enumerate(labels):
        cv2.putText(inpaint_display, label, (i*300+10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Inpainting", inpaint_display)

    # Denoising demo
    denoise_display = np.hstack([
        cv2.resize(original, (300, 200)),
        cv2.resize(noisy, (300, 200)),
        cv2.resize(denoised, (300, 200))
    ])
    labels = ["Original", "Noisy", "Denoised"]
    for i, label in enumerate(labels):
        cv2.putText(denoise_display, label, (i*300+10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Denoising", denoise_display)

    # Stylization demo
    style_display = np.hstack([
        cv2.resize(original, (200, 133)),
        cv2.resize(stylized, (200, 133)),
        cv2.resize(cv2.cvtColor(gray_pencil, cv2.COLOR_GRAY2BGR), (200, 133)),
        cv2.resize(detailed, (200, 133))
    ])
    labels = ["Original", "Stylized", "Pencil", "Detail"]
    for i, label in enumerate(labels):
        cv2.putText(style_display, label, (i*200+5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imshow("Stylization Effects", style_display)

    # HDR/Fusion demo
    fusion_display = np.hstack([
        cv2.resize(exposures[0], (200, 133)),
        cv2.resize(exposures[1], (200, 133)),
        cv2.resize(exposures[2], (200, 133)),
        cv2.resize(fusion_8bit, (200, 133))
    ])
    labels = ["Under", "Normal", "Over", "Fusion"]
    for i, label in enumerate(labels):
        cv2.putText(fusion_display, label, (i*200+5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imshow("Exposure Fusion", fusion_display)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running computational photography demonstrations...")
    print("=" * 60)
    show_demo()
