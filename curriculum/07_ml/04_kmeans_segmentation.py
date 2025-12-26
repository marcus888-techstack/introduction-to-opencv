"""
Module 9: Machine Learning - K-Means Image Segmentation
=======================================================
K-Means clustering for color-based image segmentation.

Official Docs: https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html

Topics Covered:
1. K-Means Algorithm Review
2. Color Quantization (Reducing Colors)
3. Image Segmentation by Color
4. Choosing K (Elbow Method)
5. Segmentation Applications

Real Data Used:
- fruits.jpg: Color image for segmentation
- butterfly.jpg: Natural image with distinct regions
- baboon.jpg: Complex texture image
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data.download_samples import get_sample_path

print("=" * 60)
print("Module 9: K-Means Image Segmentation")
print("=" * 60)


# =============================================================================
# 1. K-MEANS ALGORITHM REVIEW
# =============================================================================
print("\n--- 1. K-Means Algorithm ---")

kmeans_theory = """
K-Means Clustering Algorithm:

1. Initialize K cluster centroids randomly
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids as mean of assigned points
3. Return cluster assignments and centroids

For Image Segmentation:
  - Each pixel is a data point in color space (RGB or LAB)
  - K clusters represent K dominant colors
  - Pixels assigned to same cluster have similar colors

OpenCV K-Means Parameters:
  - data: Input data as float32, reshaped to (N, features)
  - K: Number of clusters
  - criteria: Termination criteria (iterations, epsilon)
  - attempts: Number of runs with different initializations
  - flags: Initialization method (KMEANS_PP_CENTERS recommended)
"""
print(kmeans_theory)


# =============================================================================
# 2. COLOR QUANTIZATION
# =============================================================================
print("\n--- 2. Color Quantization ---")


def color_quantization(image, k):
    """
    Reduce number of colors in image using K-Means.

    Args:
        image: BGR image
        k: Number of colors to keep

    Returns:
        Quantized image with only K colors
    """
    # Reshape image to (pixels, 3) for K-Means
    # Each row is a pixel's BGR values
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Apply K-Means
    compactness, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS
    )

    # Convert centers to uint8
    centers = centers.astype(np.uint8)

    # Replace each pixel with its cluster center color
    quantized = centers[labels.flatten()]

    # Reshape back to image dimensions
    quantized = quantized.reshape(image.shape)

    return quantized, labels.reshape(image.shape[:2]), centers


# Load sample image
print("Loading sample image...")
image_path = get_sample_path("fruits.jpg")
original = cv2.imread(image_path)

if original is None:
    print("Could not load fruits.jpg, trying butterfly.jpg...")
    image_path = get_sample_path("butterfly.jpg")
    original = cv2.imread(image_path)

if original is not None:
    # Resize for faster processing
    scale = 0.5
    image = cv2.resize(original, None, fx=scale, fy=scale)
    print(f"Image size: {image.shape}")

    # Quantize with different K values
    print("\nColor quantization with different K values:")
    print("-" * 40)

    for k in [2, 4, 8, 16]:
        quantized, labels, centers = color_quantization(image, k)
        unique_colors = len(np.unique(labels))
        print(f"K={k:2d}: {unique_colors} unique colors, centers shape: {centers.shape}")
else:
    print("Could not load any sample image")
    image = None


# =============================================================================
# 3. IMAGE SEGMENTATION BY COLOR
# =============================================================================
print("\n--- 3. Image Segmentation ---")


def segment_image(image, k, use_lab=True):
    """
    Segment image into K regions based on color similarity.

    Args:
        image: BGR image
        k: Number of segments
        use_lab: If True, convert to LAB color space first

    Returns:
        segmented: Colored by segment
        labels: Segment map
        masks: List of binary masks for each segment
    """
    # Optional: Convert to LAB for perceptually uniform distance
    if use_lab:
        img_for_kmeans = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        img_for_kmeans = image.copy()

    # Reshape to (pixels, 3)
    pixels = img_for_kmeans.reshape(-1, 3).astype(np.float32)

    # K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(image.shape[:2])

    # Create colored segmentation map
    # Assign distinct colors to each segment
    colors = [
        (255, 0, 0),     # Blue
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 128),   # Purple
        (0, 128, 128),   # Teal
    ]

    segmented = np.zeros_like(image)
    masks = []

    for i in range(k):
        mask = (labels == i).astype(np.uint8)
        masks.append(mask)
        color = colors[i % len(colors)]
        segmented[mask == 1] = color

    return segmented, labels, masks


if image is not None:
    print("\nSegmenting image into 4 regions...")
    segmented, seg_labels, masks = segment_image(image, k=4, use_lab=True)

    # Show segment statistics
    for i, mask in enumerate(masks):
        pixel_count = np.sum(mask)
        percentage = 100.0 * pixel_count / mask.size
        print(f"  Segment {i}: {pixel_count:,} pixels ({percentage:.1f}%)")


# =============================================================================
# 4. CHOOSING K (ELBOW METHOD)
# =============================================================================
print("\n--- 4. Elbow Method for K Selection ---")


def find_optimal_k(image, k_range=(2, 10)):
    """
    Use elbow method to find optimal K.

    Args:
        image: Input image
        k_range: Range of K values to try

    Returns:
        compactness_values: Compactness for each K
    """
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    compactness_values = []

    for k in range(k_range[0], k_range[1] + 1):
        compactness, _, _ = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        compactness_values.append(compactness)

    return list(range(k_range[0], k_range[1] + 1)), compactness_values


if image is not None:
    print("\nComputing compactness for K=2 to K=8...")
    k_values, compactness = find_optimal_k(image, k_range=(2, 8))

    print("\n  K  | Compactness | Improvement")
    print("-" * 40)
    for i, (k, c) in enumerate(zip(k_values, compactness)):
        if i > 0:
            improvement = compactness[i-1] - c
            print(f"  {k}  | {c:,.0f} | -{improvement:,.0f}")
        else:
            print(f"  {k}  | {c:,.0f} | -")

    elbow_info = """
Elbow Method:
  - Plot compactness (within-cluster variance) vs K
  - Look for "elbow" where improvement slows down
  - K at the elbow is a good choice
  - After elbow, adding clusters gives diminishing returns
"""
    print(elbow_info)


# =============================================================================
# 5. SEGMENTATION APPLICATIONS
# =============================================================================
print("\n--- 5. Practical Applications ---")

applications = """
K-Means Segmentation Applications:

1. Color Reduction / Poster Effect
   - Reduce image to N colors for artistic effect
   - Useful for compression or stylization

2. Background Removal
   - Segment into foreground/background (K=2)
   - Extract objects from uniform backgrounds

3. Object Detection Preprocessing
   - Simplify image before edge detection
   - Group similar regions for analysis

4. Medical Imaging
   - Segment tissue types in MRI/CT scans
   - Identify tumor regions

5. Satellite Imagery
   - Land use classification
   - Vegetation vs water vs urban areas
"""
print(applications)


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_segmentation():
    """Interactive visualization of K-Means segmentation."""

    if image is None:
        print("No image loaded for visualization")
        return

    def create_comparison(k):
        """Create comparison image for given K."""
        # Color quantization
        quantized, labels, centers = color_quantization(image, k)

        # Segmentation with distinct colors
        segmented, _, masks = segment_image(image, k, use_lab=True)

        # Original with segment boundaries
        boundaries = image.copy()
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(boundaries, contours, -1, (0, 255, 0), 1)

        # Stack for comparison
        row1 = np.hstack([image, quantized])
        row2 = np.hstack([segmented, boundaries])
        comparison = np.vstack([row1, row2])

        # Add labels
        h, w = image.shape[:2]
        cv2.putText(comparison, "Original", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, f"K={k} Colors", (w + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "Segments", (10, h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "Boundaries", (w + 10, h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return comparison

    # Interactive slider for K
    cv2.namedWindow("K-Means Segmentation")

    current_k = [4]  # Use list to allow modification in callback

    def on_trackbar(k):
        if k < 2:
            k = 2
        current_k[0] = k
        comparison = create_comparison(k)
        cv2.imshow("K-Means Segmentation", comparison)

    cv2.createTrackbar("K", "K-Means Segmentation", 4, 10, on_trackbar)

    # Initial display
    comparison = create_comparison(4)
    cv2.imshow("K-Means Segmentation", comparison)

    print("\nUse slider to change K value")
    print("Press 'q' to quit")

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def visualize_elbow():
    """Visualize the elbow method."""

    if image is None:
        return

    k_values, compactness = find_optimal_k(image, k_range=(2, 8))

    # Normalize compactness for visualization
    max_comp = max(compactness)
    min_comp = min(compactness)

    # Create visualization
    viz_width = 400
    viz_height = 300
    viz = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
    viz[:] = (255, 255, 255)

    # Draw axes
    margin = 50
    cv2.line(viz, (margin, viz_height - margin), (viz_width - margin, viz_height - margin), (0, 0, 0), 2)
    cv2.line(viz, (margin, margin), (margin, viz_height - margin), (0, 0, 0), 2)

    # Plot points and lines
    x_scale = (viz_width - 2 * margin) / (len(k_values) - 1)
    y_scale = (viz_height - 2 * margin) / (max_comp - min_comp + 1)

    points = []
    for i, (k, c) in enumerate(zip(k_values, compactness)):
        x = int(margin + i * x_scale)
        y = int(viz_height - margin - (c - min_comp) * y_scale)
        points.append((x, y))

        # Draw point
        cv2.circle(viz, (x, y), 5, (0, 0, 255), -1)

        # K label
        cv2.putText(viz, str(k), (x - 5, viz_height - margin + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw lines connecting points
    for i in range(len(points) - 1):
        cv2.line(viz, points[i], points[i+1], (255, 0, 0), 2)

    # Labels
    cv2.putText(viz, "Compactness", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(viz, "K (clusters)", (viz_width - 100, viz_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(viz, "Elbow Method", (viz_width // 2 - 50, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Elbow Method", viz)


def demo_background_removal():
    """Demo background removal using K-Means."""

    if image is None:
        return

    # Simple K=2 segmentation for background/foreground
    quantized, labels, centers = color_quantization(image, k=2)

    # Determine which cluster is likely background (larger area usually)
    unique, counts = np.unique(labels, return_counts=True)
    bg_label = unique[np.argmax(counts)]

    # Create mask (foreground)
    mask = (labels != bg_label).astype(np.uint8) * 255

    # Apply mask to original
    foreground = cv2.bitwise_and(image, image, mask=mask)

    # Create comparison
    comparison = np.hstack([image, foreground])
    cv2.putText(comparison, "Original", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(comparison, "Background Removed (K=2)", (image.shape[1] + 10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Background Removal", comparison)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Visualizing K-Means Segmentation...")
    print("=" * 60)

    if image is not None:
        # Show elbow method
        visualize_elbow()

        # Show background removal
        demo_background_removal()

        print("\nPress any key to start interactive demo...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Interactive segmentation
        visualize_segmentation()
    else:
        print("Could not load sample images for visualization")
