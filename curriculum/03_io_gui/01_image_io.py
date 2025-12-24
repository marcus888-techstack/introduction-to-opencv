"""
Module 3: I/O and GUI - Reading and Writing Images
===================================================
Loading, saving, and displaying images with OpenCV.

Official Docs: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html

Topics Covered:
1. Reading Images (imread)
2. Writing Images (imwrite)
3. Image Formats
4. Display Windows (imshow)
5. Handling Paths and Errors
"""

import cv2
import numpy as np
import os

print("=" * 60)
print("Module 3: Reading and Writing Images")
print("=" * 60)


# =============================================================================
# 1. READING IMAGES (imread)
# =============================================================================
print("\n--- 1. Reading Images ---")

# Create a sample image for demos
sample_img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(sample_img, (50, 50), (350, 250), (0, 255, 0), -1)
cv2.circle(sample_img, (200, 150), 80, (255, 0, 0), -1)
cv2.putText(sample_img, "Sample", (130, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Save sample for testing
cv2.imwrite("sample_test.png", sample_img)

# Read in color (default)
img_color = cv2.imread("sample_test.png", cv2.IMREAD_COLOR)
print(f"Color image shape: {img_color.shape}")  # (H, W, 3)

# Read in grayscale
img_gray = cv2.imread("sample_test.png", cv2.IMREAD_GRAYSCALE)
print(f"Grayscale image shape: {img_gray.shape}")  # (H, W)

# Read with alpha channel (if exists)
img_unchanged = cv2.imread("sample_test.png", cv2.IMREAD_UNCHANGED)
print(f"Unchanged image shape: {img_unchanged.shape}")

# Read flags summary:
# cv2.IMREAD_COLOR      = 1  (default, ignore transparency)
# cv2.IMREAD_GRAYSCALE  = 0  (convert to grayscale)
# cv2.IMREAD_UNCHANGED  = -1 (include alpha channel)


# =============================================================================
# 2. HANDLING ERRORS AND PATHS
# =============================================================================
print("\n--- 2. Handling Errors ---")

# Check if file exists before reading
file_path = "nonexistent.jpg"
if not os.path.exists(file_path):
    print(f"File does not exist: {file_path}")

# imread returns None if file cannot be read
result = cv2.imread("nonexistent.jpg")
if result is None:
    print("Failed to load image - imread returned None")

# Always check before using
def load_image_safely(path):
    """Load image with error checking."""
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not decode image: {path}")
        return None

    return img


# =============================================================================
# 3. WRITING IMAGES (imwrite)
# =============================================================================
print("\n--- 3. Writing Images ---")

# Basic save
success = cv2.imwrite("output.png", sample_img)
print(f"Save PNG successful: {success}")

# Save with different formats
cv2.imwrite("output.jpg", sample_img)
cv2.imwrite("output.bmp", sample_img)

# JPEG quality (0-100, default 95)
cv2.imwrite("output_low.jpg", sample_img, [cv2.IMWRITE_JPEG_QUALITY, 30])
cv2.imwrite("output_high.jpg", sample_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

# PNG compression (0-9, higher = smaller file, slower)
cv2.imwrite("output_compressed.png", sample_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# Check file sizes
import os
for f in ["output.jpg", "output_low.jpg", "output_high.jpg"]:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"{f}: {size} bytes")


# =============================================================================
# 4. IMAGE FORMATS
# =============================================================================
print("\n--- 4. Supported Image Formats ---")

formats = """
Common formats supported by OpenCV:
  - PNG  (.png)  - Lossless, supports transparency
  - JPEG (.jpg)  - Lossy, small files, no transparency
  - BMP  (.bmp)  - Uncompressed, large files
  - TIFF (.tiff) - Flexible, professional use
  - WebP (.webp) - Modern, good compression
  - PBM/PGM/PPM  - Simple ASCII formats
"""
print(formats)


# =============================================================================
# 5. ENCODING AND DECODING IN MEMORY
# =============================================================================
print("\n--- 5. In-Memory Encoding ---")

# Encode image to bytes (for network transmission, etc.)
success, encoded = cv2.imencode('.png', sample_img)
print(f"Encoded to {len(encoded)} bytes")

# Decode bytes back to image
decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
print(f"Decoded shape: {decoded.shape}")

# Encode as JPEG with specific quality
success, jpg_bytes = cv2.imencode('.jpg', sample_img,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
print(f"JPEG encoded: {len(jpg_bytes)} bytes")


# =============================================================================
# 6. WORKING WITH NUMPY ARRAYS
# =============================================================================
print("\n--- 6. NumPy Array Operations ---")

# Save numpy array directly
np_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
cv2.imwrite("random.png", np_array)

# Load and convert data types
img = cv2.imread("random.png")
print(f"Loaded dtype: {img.dtype}")  # uint8

# Convert to float for processing
img_float = img.astype(np.float32) / 255.0
print(f"Float range: {img_float.min():.2f} - {img_float.max():.2f}")

# Convert back for saving
img_back = (img_float * 255).astype(np.uint8)
cv2.imwrite("converted.png", img_back)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display image I/O demo."""

    # Show different read modes
    color = cv2.imread("sample_test.png", cv2.IMREAD_COLOR)
    gray = cv2.imread("sample_test.png", cv2.IMREAD_GRAYSCALE)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    display = np.hstack([color, gray_bgr])
    cv2.putText(display, "Color", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, "Grayscale", (410, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Image Loading Modes", display)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Clean up test files
    for f in ["sample_test.png", "output.png", "output.jpg", "output.bmp",
              "output_low.jpg", "output_high.jpg", "output_compressed.png",
              "random.png", "converted.png"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running image I/O demonstrations...")
    print("=" * 60)
    show_demo()
