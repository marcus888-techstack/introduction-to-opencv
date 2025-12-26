"""
Module 2: Image Processing - Filtering
=======================================
Smoothing, sharpening, and filtering operations.

Official Docs: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html

Topics Covered:
1. Averaging/Box Filter
2. Gaussian Blur
3. Median Filter
4. Bilateral Filter
5. Custom Kernels
6. Sharpening
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 2: Image Processing - Filtering")
print("=" * 60)


def load_and_add_noise(filename="lena.jpg"):
    """Load real image and add noise for filter testing."""
    # Try to load sample image
    img = get_image(filename)

    if img is None:
        # Try alternative images
        for alt in ["baboon.jpg", "fruits.jpg", "building.jpg"]:
            img = get_image(alt)
            if img is not None:
                print(f"Using sample image: {alt}")
                break

    if img is None:
        # Final fallback: create synthetic image
        print("No sample images found. Using synthetic image.")
        print("Run: python curriculum/sample_data/download_samples.py")
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        for i in range(400):
            img[:, i] = (i * 255 // 400, 128, 255 - i * 255 // 400)
        cv2.circle(img, (200, 150), 60, (0, 255, 255), -1)
    else:
        print(f"Using sample image: {filename}")
        # Resize for consistent demo
        img = cv2.resize(img, (400, 300))

    # Add salt and pepper noise
    noise = np.random.random(img.shape[:2])
    img[noise < 0.02] = 0      # Pepper
    img[noise > 0.98] = 255    # Salt

    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + gaussian_noise, 0, 255).astype(np.uint8)

    return img


# Load test image with noise
original = load_and_add_noise()


# =============================================================================
# 1. AVERAGING (BOX FILTER)
# =============================================================================
print("\n--- 1. Averaging (Box Filter) ---")

# Simple averaging
blur_3x3 = cv2.blur(original, (3, 3))
blur_5x5 = cv2.blur(original, (5, 5))
blur_9x9 = cv2.blur(original, (9, 9))

print("Box filter: Each pixel = average of neighbors")
print("Larger kernel = more blur")


# =============================================================================
# 2. GAUSSIAN BLUR
# =============================================================================
print("\n--- 2. Gaussian Blur ---")

# Gaussian blur (weighted average, center has more weight)
gaussian_3 = cv2.GaussianBlur(original, (3, 3), 0)
gaussian_5 = cv2.GaussianBlur(original, (5, 5), 0)
gaussian_9 = cv2.GaussianBlur(original, (9, 9), 0)

# With specific sigma
gaussian_sigma = cv2.GaussianBlur(original, (0, 0), sigmaX=3)

print("Gaussian: Bell-curve weighted average")
print("Better edge preservation than box filter")


# =============================================================================
# 3. MEDIAN FILTER
# =============================================================================
print("\n--- 3. Median Filter ---")

# Median filter (great for salt-and-pepper noise)
median_3 = cv2.medianBlur(original, 3)
median_5 = cv2.medianBlur(original, 5)
median_9 = cv2.medianBlur(original, 9)

print("Median: Replaces pixel with median of neighbors")
print("Best for salt-and-pepper noise!")


# =============================================================================
# 4. BILATERAL FILTER
# =============================================================================
print("\n--- 4. Bilateral Filter ---")

# Bilateral filter (edge-preserving smoothing)
# Parameters: d (diameter), sigmaColor, sigmaSpace
bilateral = cv2.bilateralFilter(original, 9, 75, 75)
bilateral_strong = cv2.bilateralFilter(original, 15, 100, 100)

print("Bilateral: Smooths while preserving edges")
print("Great for face smoothing/beautification")


# =============================================================================
# 5. CUSTOM KERNELS
# =============================================================================
print("\n--- 5. Custom Kernels ---")

# Identity kernel (no change)
kernel_identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)

# Edge detection kernel (Laplacian-like)
kernel_edge = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Emboss kernel
kernel_emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=np.float32)

# Apply custom kernels
identity_result = cv2.filter2D(original, -1, kernel_identity)
edge_result = cv2.filter2D(original, -1, kernel_edge)
emboss_result = cv2.filter2D(original, -1, kernel_emboss)

print("Custom kernels applied: identity, edge, emboss")


# =============================================================================
# 6. SHARPENING
# =============================================================================
print("\n--- 6. Sharpening ---")

# Sharpening kernel
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# Strong sharpening
kernel_sharpen_strong = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)

sharpened = cv2.filter2D(original, -1, kernel_sharpen)
sharpened_strong = cv2.filter2D(original, -1, kernel_sharpen_strong)

# Unsharp masking (another sharpening technique)
gaussian = cv2.GaussianBlur(original, (0, 0), 3)
unsharp_mask = cv2.addWeighted(original, 1.5, gaussian, -0.5, 0)

print("Sharpening enhances edges and details")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display demo of all filters."""

    # Blur comparison
    blur_row = np.hstack([original, blur_3x3, blur_5x5, gaussian_5])
    blur_row = cv2.resize(blur_row, (1200, 225))

    labels = ["Original", "Box 3x3", "Box 5x5", "Gaussian 5x5"]
    for i, label in enumerate(labels):
        cv2.putText(blur_row, label, (i * 300 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Blur Comparison", blur_row)

    # Noise removal comparison
    noise_row = np.hstack([original, median_5, bilateral, gaussian_5])
    noise_row = cv2.resize(noise_row, (1200, 225))

    labels = ["Noisy", "Median", "Bilateral", "Gaussian"]
    for i, label in enumerate(labels):
        cv2.putText(noise_row, label, (i * 300 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Noise Removal", noise_row)

    # Effects
    effects_row = np.hstack([original, sharpened, emboss_result, edge_result])
    effects_row = cv2.resize(effects_row, (1200, 225))

    labels = ["Original", "Sharpened", "Emboss", "Edges"]
    for i, label in enumerate(labels):
        cv2.putText(effects_row, label, (i * 300 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Effects", effects_row)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running filter demonstrations...")
    print("=" * 60)
    show_demo()
