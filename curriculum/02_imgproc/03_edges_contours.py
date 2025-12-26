"""
Module 2: Image Processing - Edges and Contours
================================================
Edge detection and contour finding/analysis.

Topics Covered:
1. Sobel Derivatives
2. Laplacian
3. Canny Edge Detection
4. Finding Contours
5. Contour Properties
6. Contour Approximation
7. Bounding Shapes
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 2: Edges and Contours")
print("=" * 60)


def load_edge_image():
    """Load a real image for edge detection, or create one if not available."""
    # Try to load sample images good for edge detection
    for sample in ["building.jpg", "sudoku.png", "lena.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            # Resize for consistent display
            img = cv2.resize(img, (600, 400))
            return img

    # Fallback to synthetic
    print("No sample image found. Using synthetic shapes.")
    print("Run: python curriculum/sample_data/download_samples.py")
    return create_test_image()


def create_test_image():
    """Create image with shapes for edge/contour testing (fallback)."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    # Draw various shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img, (250, 100), 50, (255, 255, 255), -1)
    cv2.ellipse(img, (400, 100), (60, 40), 30, 0, 360, (255, 255, 255), -1)

    # Triangle
    pts = np.array([[100, 250], [50, 350], [150, 350]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # Star-like shape
    pts = np.array([[300, 200], [350, 280], [280, 240],
                    [320, 320], [270, 270], [220, 320],
                    [250, 240], [180, 280], [230, 200], [270, 280]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # Irregular polygon
    pts = np.array([[450, 200], [550, 220], [530, 350], [420, 330]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    return img


original = load_edge_image()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)


# =============================================================================
# 1. SOBEL DERIVATIVES
# =============================================================================
print("\n--- 1. Sobel Derivatives ---")

# Sobel in X direction (vertical edges)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

# Sobel in Y direction (horizontal edges)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combined
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

print("Sobel: First derivative, detects edges in X or Y direction")


# =============================================================================
# 2. LAPLACIAN
# =============================================================================
print("\n--- 2. Laplacian ---")

# Laplacian (second derivative)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

print("Laplacian: Second derivative, detects all edges")


# =============================================================================
# 3. CANNY EDGE DETECTION
# =============================================================================
print("\n--- 3. Canny Edge Detection ---")

# Canny with different thresholds
canny_low = cv2.Canny(gray, 50, 100)
canny_mid = cv2.Canny(gray, 100, 200)
canny_high = cv2.Canny(gray, 150, 300)

print("Canny: Multi-stage edge detector")
print("  - Low threshold: More edges (noisy)")
print("  - High threshold: Fewer edges (clean)")


# =============================================================================
# 4. FINDING CONTOURS
# =============================================================================
print("\n--- 4. Finding Contours ---")

# Find contours on edge image
edges = cv2.Canny(gray, 100, 200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours")

# Draw all contours
contour_img = original.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Retrieval modes:
# RETR_EXTERNAL - only outermost contours
# RETR_LIST - all contours, no hierarchy
# RETR_TREE - full hierarchy

# Approximation methods:
# CHAIN_APPROX_NONE - all points
# CHAIN_APPROX_SIMPLE - compress segments


# =============================================================================
# 5. CONTOUR PROPERTIES
# =============================================================================
print("\n--- 5. Contour Properties ---")

properties_img = original.copy()

for i, contour in enumerate(contours):
    # Area
    area = cv2.contourArea(contour)

    # Perimeter
    perimeter = cv2.arcLength(contour, True)

    # Centroid (using moments)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Draw centroid
    cv2.circle(properties_img, (cx, cy), 5, (0, 0, 255), -1)

    print(f"Contour {i}: Area={area:.0f}, Perimeter={perimeter:.1f}, "
          f"Centroid=({cx},{cy})")


# =============================================================================
# 6. CONTOUR APPROXIMATION
# =============================================================================
print("\n--- 6. Contour Approximation ---")

approx_img = original.copy()

for contour in contours:
    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Draw approximated contour
    cv2.drawContours(approx_img, [approx], -1, (255, 0, 0), 2)

    # Draw vertices
    for point in approx:
        cv2.circle(approx_img, tuple(point[0]), 4, (0, 255, 255), -1)

    print(f"Original: {len(contour)} points -> Approximated: {len(approx)} points")


# =============================================================================
# 7. BOUNDING SHAPES
# =============================================================================
print("\n--- 7. Bounding Shapes ---")

bounds_img = original.copy()

for contour in contours:
    if cv2.contourArea(contour) < 500:
        continue

    # Bounding rectangle (axis-aligned)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(bounds_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Rotated rectangle (minimum area)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(bounds_img, [box], 0, (255, 0, 0), 2)

    # Minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(bounds_img, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)

    # Fitted ellipse (needs >= 5 points)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(bounds_img, ellipse, (255, 0, 255), 2)

print("Green: Bounding rect, Blue: Rotated rect")
print("Yellow: Min circle, Magenta: Fitted ellipse")


# =============================================================================
# 8. SHAPE DETECTION
# =============================================================================
print("\n--- 8. Shape Detection ---")

shapes_img = original.copy()

for contour in contours:
    if cv2.contourArea(contour) < 500:
        continue

    # Approximate to polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    # Get centroid for label
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        continue

    # Classify shape
    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        # Check if square or rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h)
        shape = "Square" if 0.9 <= aspect <= 1.1 else "Rectangle"
    elif vertices == 5:
        shape = "Pentagon"
    elif vertices > 5:
        # Check circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        shape = "Circle" if circularity > 0.8 else "Polygon"
    else:
        shape = "Unknown"

    cv2.putText(shapes_img, shape, (cx - 30, cy),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

print("Shapes detected based on vertex count and circularity")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display edge and contour demos."""

    # Edge detection comparison
    edge_row = np.hstack([gray, sobel_x, sobel_y, laplacian])
    edge_row = cv2.resize(edge_row, (1200, 200))

    labels = ["Original", "Sobel X", "Sobel Y", "Laplacian"]
    for i, label in enumerate(labels):
        cv2.putText(edge_row, label, (i * 300 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 2)

    cv2.imshow("Edge Detection", edge_row)

    # Canny thresholds
    canny_row = np.hstack([gray, canny_low, canny_mid, canny_high])
    canny_row = cv2.resize(canny_row, (1200, 200))

    labels = ["Original", "Canny Low", "Canny Mid", "Canny High"]
    for i, label in enumerate(labels):
        cv2.putText(canny_row, label, (i * 300 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 2)

    cv2.imshow("Canny Thresholds", canny_row)

    # Contours
    cv2.imshow("Contours Found", contour_img)
    cv2.imshow("Bounding Shapes", bounds_img)
    cv2.imshow("Shape Detection", shapes_img)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running edge and contour demonstrations...")
    print("=" * 60)
    show_demo()
