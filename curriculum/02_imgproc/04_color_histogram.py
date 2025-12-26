"""
Module 2: Image Processing - Color Spaces and Histograms
=========================================================
Color conversions, histograms, and histogram operations.

Topics Covered:
1. Color Space Conversions
2. Color Channel Splitting
3. Histogram Calculation
4. Histogram Equalization
5. CLAHE (Adaptive Equalization)
6. Histogram Comparison
7. Color-based Segmentation
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 2: Color Spaces and Histograms")
print("=" * 60)


def load_color_image():
    """Load a real colorful image, or create one if not available."""
    # Try to load colorful sample images
    for sample in ["fruits.jpg", "lena.jpg", "baboon.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            # Resize for consistent display
            img = cv2.resize(img, (400, 300))
            return img

    # Fallback to synthetic
    print("No color sample found. Using synthetic image.")
    print("Run: python curriculum/sample_data/download_samples.py")
    return create_color_image()


def create_color_image():
    """Create a colorful test image (fallback)."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Gradient background
    for i in range(400):
        img[:, i, 0] = int(255 * i / 400)  # Blue gradient
        img[:, i, 2] = int(255 * (400 - i) / 400)  # Red gradient

    # Add colored shapes
    cv2.circle(img, (100, 150), 60, (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (200, 80), (300, 180), (255, 255, 0), -1)  # Cyan
    cv2.circle(img, (350, 200), 50, (0, 0, 255), -1)  # Red
    cv2.rectangle(img, (50, 200), (150, 280), (255, 0, 255), -1)  # Magenta

    return img


original = load_color_image()


# =============================================================================
# 1. COLOR SPACE CONVERSIONS
# =============================================================================
print("\n--- 1. Color Space Conversions ---")

# BGR to Grayscale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# BGR to RGB (for matplotlib compatibility)
rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# BGR to HSV (Hue, Saturation, Value)
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

# BGR to LAB
lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

# BGR to YCrCb
ycrcb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)

print("Common color spaces:")
print("  BGR - Blue, Green, Red (OpenCV default)")
print("  RGB - Red, Green, Blue (display standard)")
print("  HSV - Hue, Saturation, Value (color picking)")
print("  LAB - Lightness, a, b (perceptual uniformity)")
print("  YCrCb - Luminance, Chroma (video)")


# =============================================================================
# 2. HSV COLOR SPACE
# =============================================================================
print("\n--- 2. HSV Color Space ---")

# Split HSV channels
h, s, v = cv2.split(hsv)

print("HSV channels:")
print(f"  Hue: 0-180 (color type)")
print(f"  Saturation: 0-255 (color intensity)")
print(f"  Value: 0-255 (brightness)")

# Common HSV ranges for colors:
# Red: H = 0-10 or 160-180
# Orange: H = 10-25
# Yellow: H = 25-35
# Green: H = 35-85
# Blue: H = 85-130
# Purple: H = 130-160


# =============================================================================
# 3. HISTOGRAM CALCULATION
# =============================================================================
print("\n--- 3. Histogram Calculation ---")

# Grayscale histogram
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Color histograms
hist_b = cv2.calcHist([original], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([original], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([original], [2], None, [256], [0, 256])

print(f"Histogram shape: {hist_gray.shape}")
print("Histogram shows distribution of pixel intensities")


def draw_histogram(hist, color, height=200):
    """Draw a simple histogram visualization."""
    h = np.zeros((height, 256, 3), dtype=np.uint8)
    hist = hist.flatten()
    hist = hist / hist.max() * height  # Normalize

    for i, val in enumerate(hist):
        cv2.line(h, (i, height), (i, height - int(val)), color, 1)

    return h


# =============================================================================
# 4. HISTOGRAM EQUALIZATION
# =============================================================================
print("\n--- 4. Histogram Equalization ---")

# Create low contrast image
low_contrast = cv2.convertScaleAbs(gray, alpha=0.5, beta=50)

# Apply histogram equalization
equalized = cv2.equalizeHist(low_contrast)

print("Histogram Equalization: Spreads pixel values across full range")
print("Result: Improved contrast")


# =============================================================================
# 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# =============================================================================
print("\n--- 5. CLAHE ---")

# Standard equalization can over-amplify noise
# CLAHE divides image into tiles and equalizes each

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(low_contrast)

# Different clip limits
clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
clahe_high = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

clahe_low_result = clahe_low.apply(low_contrast)
clahe_high_result = clahe_high.apply(low_contrast)

print("CLAHE: Adaptive equalization with contrast limiting")
print("clipLimit: Controls contrast amplification (lower = less noise)")


# =============================================================================
# 6. COLOR-BASED SEGMENTATION
# =============================================================================
print("\n--- 6. Color-based Segmentation (HSV) ---")

# Detect green color
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
green_only = cv2.bitwise_and(original, original, mask=mask_green)

# Detect red color (wraps around in HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
red_only = cv2.bitwise_and(original, original, mask=mask_red)

# Detect blue color
lower_blue = np.array([85, 100, 100])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
blue_only = cv2.bitwise_and(original, original, mask=mask_blue)

print("Color segmentation using HSV thresholds")


# =============================================================================
# 7. BACKPROJECTION
# =============================================================================
print("\n--- 7. Histogram Backprojection ---")

# Select a region of interest (green circle area)
roi = original[90:210, 40:160]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Calculate histogram of ROI
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Backproject to find similar colors
backproj = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

print("Backprojection: Find pixels similar to ROI histogram")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display color and histogram demos."""

    # Color spaces
    h_display = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
    s_display = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    v_display = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    color_row = np.hstack([original, h_display, s_display, v_display])
    color_row = cv2.resize(color_row, (1200, 225))

    labels = ["Original BGR", "Hue", "Saturation", "Value"]
    for i, label in enumerate(labels):
        cv2.putText(color_row, label, (i * 300 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("HSV Color Space", color_row)

    # Histogram visualization
    hist_b_img = draw_histogram(hist_b, (255, 0, 0))
    hist_g_img = draw_histogram(hist_g, (0, 255, 0))
    hist_r_img = draw_histogram(hist_r, (0, 0, 255))

    hist_display = np.hstack([hist_b_img, hist_g_img, hist_r_img])
    cv2.imshow("Color Histograms (B, G, R)", hist_display)

    # Equalization comparison
    eq_row = np.hstack([
        cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(clahe_result, cv2.COLOR_GRAY2BGR)
    ])
    eq_row = cv2.resize(eq_row, (900, 225))

    labels = ["Low Contrast", "Equalized", "CLAHE"]
    for i, label in enumerate(labels):
        cv2.putText(eq_row, label, (i * 300 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Histogram Equalization", eq_row)

    # Color segmentation
    seg_row = np.hstack([original, green_only, red_only, blue_only])
    seg_row = cv2.resize(seg_row, (1200, 225))

    labels = ["Original", "Green Only", "Red Only", "Blue Only"]
    for i, label in enumerate(labels):
        cv2.putText(seg_row, label, (i * 300 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Color Segmentation", seg_row)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running color and histogram demonstrations...")
    print("=" * 60)
    show_demo()
