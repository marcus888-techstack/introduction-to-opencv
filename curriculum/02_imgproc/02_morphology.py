"""
Module 2: Image Processing - Morphological Operations
======================================================
Erosion, dilation, opening, closing, and advanced morphology.

Official Docs: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

Topics Covered:
1. Erosion
2. Dilation
3. Opening (erosion + dilation)
4. Closing (dilation + erosion)
5. Gradient
6. Top Hat / Black Hat
7. Structuring Elements
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 2: Morphological Operations")
print("=" * 60)


def create_test_image():
    """Create image with text and shapes for morphology demo."""
    img = np.zeros((300, 500), dtype=np.uint8)

    # Add text
    cv2.putText(img, "OpenCV", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

    # Add shapes
    cv2.circle(img, (400, 80), 40, 255, -1)
    cv2.rectangle(img, (350, 150), (450, 250), 255, -1)

    # Add some noise (small dots)
    noise = np.random.random(img.shape) > 0.995
    img[noise] = 255

    # Add some gaps (small black dots)
    gaps = np.random.random(img.shape) > 0.998
    img[gaps] = 0

    return img


original = create_test_image()


# =============================================================================
# 1. STRUCTURING ELEMENTS (KERNELS)
# =============================================================================
print("\n--- 1. Structuring Elements ---")

# Rectangle kernel
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print("Rectangle kernel:\n", kernel_rect)

# Ellipse kernel
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print("\nEllipse kernel:\n", kernel_ellipse)

# Cross kernel
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
print("\nCross kernel:\n", kernel_cross)

# Use 3x3 kernel for demos
kernel = np.ones((3, 3), np.uint8)


# =============================================================================
# 2. EROSION
# =============================================================================
print("\n--- 2. Erosion ---")

# Erosion shrinks white regions (removes pixels at boundaries)
erosion_1 = cv2.erode(original, kernel, iterations=1)
erosion_2 = cv2.erode(original, kernel, iterations=2)
erosion_3 = cv2.erode(original, kernel, iterations=3)

print("Erosion: Shrinks white regions, removes noise")
print("Use: Remove small white noise, separate connected objects")


# =============================================================================
# 3. DILATION
# =============================================================================
print("\n--- 3. Dilation ---")

# Dilation expands white regions (adds pixels at boundaries)
dilation_1 = cv2.dilate(original, kernel, iterations=1)
dilation_2 = cv2.dilate(original, kernel, iterations=2)
dilation_3 = cv2.dilate(original, kernel, iterations=3)

print("Dilation: Expands white regions, fills gaps")
print("Use: Fill small holes, connect nearby objects")


# =============================================================================
# 4. OPENING (Erosion followed by Dilation)
# =============================================================================
print("\n--- 4. Opening ---")

# Opening removes small white noise
opening = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
opening_large = cv2.morphologyEx(original, cv2.MORPH_OPEN,
                                  np.ones((5, 5), np.uint8))

print("Opening = Erosion + Dilation")
print("Use: Remove small white spots (noise) while preserving shape")


# =============================================================================
# 5. CLOSING (Dilation followed by Erosion)
# =============================================================================
print("\n--- 5. Closing ---")

# Closing fills small black holes
closing = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
closing_large = cv2.morphologyEx(original, cv2.MORPH_CLOSE,
                                  np.ones((5, 5), np.uint8))

print("Closing = Dilation + Erosion")
print("Use: Fill small black holes while preserving shape")


# =============================================================================
# 6. MORPHOLOGICAL GRADIENT
# =============================================================================
print("\n--- 6. Morphological Gradient ---")

# Gradient = Dilation - Erosion (gives outline)
gradient = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel)

print("Gradient = Dilation - Erosion")
print("Use: Find edges/boundaries of objects")


# =============================================================================
# 7. TOP HAT and BLACK HAT
# =============================================================================
print("\n--- 7. Top Hat and Black Hat ---")

# Top Hat = Original - Opening (bright spots smaller than kernel)
tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT,
                          np.ones((15, 15), np.uint8))

# Black Hat = Closing - Original (dark spots smaller than kernel)
blackhat = cv2.morphologyEx(original, cv2.MORPH_BLACKHAT,
                            np.ones((15, 15), np.uint8))

print("Top Hat: Highlights bright spots smaller than kernel")
print("Black Hat: Highlights dark spots smaller than kernel")


# =============================================================================
# 8. PRACTICAL EXAMPLE: Text Cleaning
# =============================================================================
print("\n--- 8. Practical: Text Cleaning ---")

# Simulate dirty scanned text
dirty_text = original.copy()

# Add random noise
noise = np.random.random(dirty_text.shape) > 0.97
dirty_text[noise] = 255

# Clean using opening
cleaned_text = cv2.morphologyEx(dirty_text, cv2.MORPH_OPEN, kernel)

# Then close to fill gaps
cleaned_text = cv2.morphologyEx(cleaned_text, cv2.MORPH_CLOSE, kernel)

print("Cleaned noisy text using Opening + Closing")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display morphological operations demo."""

    # Erosion demo
    erosion_row = np.hstack([original, erosion_1, erosion_2, erosion_3])
    erosion_row = cv2.resize(erosion_row, (1000, 150))

    labels = ["Original", "Erode x1", "Erode x2", "Erode x3"]
    for i, label in enumerate(labels):
        cv2.putText(erosion_row, label, (i * 250 + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

    cv2.imshow("Erosion", erosion_row)

    # Dilation demo
    dilation_row = np.hstack([original, dilation_1, dilation_2, dilation_3])
    dilation_row = cv2.resize(dilation_row, (1000, 150))

    labels = ["Original", "Dilate x1", "Dilate x2", "Dilate x3"]
    for i, label in enumerate(labels):
        cv2.putText(dilation_row, label, (i * 250 + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

    cv2.imshow("Dilation", dilation_row)

    # Open/Close demo
    oc_row = np.hstack([original, opening, closing, gradient])
    oc_row = cv2.resize(oc_row, (1000, 150))

    labels = ["Original", "Opening", "Closing", "Gradient"]
    for i, label in enumerate(labels):
        cv2.putText(oc_row, label, (i * 250 + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

    cv2.imshow("Opening / Closing / Gradient", oc_row)

    # Top Hat / Black Hat
    hat_row = np.hstack([original, tophat, blackhat])
    hat_row = cv2.resize(hat_row, (750, 150))

    labels = ["Original", "Top Hat", "Black Hat"]
    for i, label in enumerate(labels):
        cv2.putText(hat_row, label, (i * 250 + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

    cv2.imshow("Top Hat / Black Hat", hat_row)

    # Text cleaning
    text_row = np.hstack([dirty_text, cleaned_text])
    text_row = cv2.resize(text_row, (500, 150))

    cv2.putText(text_row, "Dirty", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
    cv2.putText(text_row, "Cleaned", (260, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

    cv2.imshow("Text Cleaning", text_row)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running morphology demonstrations...")
    print("=" * 60)
    show_demo()
