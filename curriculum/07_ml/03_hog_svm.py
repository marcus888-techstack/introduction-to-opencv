"""
Module 9: Machine Learning - HOG + SVM Classification
=====================================================
Histogram of Oriented Gradients (HOG) feature extraction with SVM.

Official Docs:
- HOG: https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html
- SVM: https://docs.opencv.org/4.x/d1/d73/tutorial_introduction_to_svm.html

Topics Covered:
1. HOG Feature Extraction
2. HOG Visualization
3. HOG Parameters and Effects
4. HOG + SVM for Digit Classification
5. Pedestrian Detection with Built-in HOG

Real Data Used:
- digits.png: Handwritten digits for classification
- vtest.avi: Pedestrian video for detection demo
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data.download_samples import get_sample_path

print("=" * 60)
print("Module 9: HOG Features + SVM Classification")
print("=" * 60)


# =============================================================================
# 1. HOG FEATURE EXTRACTION
# =============================================================================
print("\n--- 1. HOG Feature Extraction ---")

hog_theory = """
HOG (Histogram of Oriented Gradients):
  - Captures edge/gradient structure of an image
  - Divides image into cells and blocks
  - Computes gradient histogram for each cell
  - Normalizes across blocks for lighting invariance

HOG Parameters:
  - winSize: Detection window size
  - blockSize: Block size for normalization
  - blockStride: Step between blocks
  - cellSize: Cell size for histogram
  - nbins: Number of gradient direction bins (typically 9)
"""
print(hog_theory)

# Create HOG descriptor for 20x20 digit images
# Parameters optimized for small images
hog = cv2.HOGDescriptor(
    _winSize=(20, 20),      # Window size (same as digit)
    _blockSize=(10, 10),    # Block size
    _blockStride=(5, 5),    # Block stride
    _cellSize=(5, 5),       # Cell size
    _nbins=9                # Number of orientation bins
)

# Calculate expected feature vector size
# Each block produces cellsPerBlock * cellsPerBlock * nbins features
# Number of blocks = ((winSize - blockSize) / blockStride + 1)^2
n_blocks_x = (20 - 10) // 5 + 1  # 3
n_blocks_y = (20 - 10) // 5 + 1  # 3
cells_per_block = (10 // 5) ** 2  # 4 cells per block (2x2)
n_bins = 9
expected_features = n_blocks_x * n_blocks_y * cells_per_block * n_bins

print(f"HOG Configuration for 20x20 images:")
print(f"  Blocks: {n_blocks_x} x {n_blocks_y} = {n_blocks_x * n_blocks_y}")
print(f"  Cells per block: {cells_per_block}")
print(f"  Orientation bins: {n_bins}")
print(f"  Feature vector size: {expected_features}")


# =============================================================================
# 2. HOG VISUALIZATION
# =============================================================================
print("\n--- 2. HOG Visualization ---")


def visualize_hog(image, hog_descriptor, cell_size=5, scale=3):
    """
    Visualize HOG features on an image.

    Args:
        image: Input image
        hog_descriptor: HOG descriptor object
        cell_size: Size of HOG cells
        scale: Visualization scale factor
    """
    # Compute gradients
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

    # Compute magnitude and angle
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Create visualization
    h, w = image.shape
    viz = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

    # Draw gradient lines for each cell
    n_cells_x = w // cell_size
    n_cells_y = h // cell_size

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            # Cell boundaries
            x1 = cx * cell_size
            y1 = cy * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            # Average gradient in cell
            cell_mag = mag[y1:y2, x1:x2].mean()
            cell_angle = angle[y1:y2, x1:x2].mean()

            # Center of cell (scaled)
            center_x = int((x1 + cell_size/2) * scale)
            center_y = int((y1 + cell_size/2) * scale)

            # Draw gradient direction
            length = int(cell_mag * scale / 30)  # Normalize length
            if length > 1:
                rad = np.radians(cell_angle)
                dx = int(length * np.cos(rad))
                dy = int(length * np.sin(rad))
                cv2.line(viz, (center_x - dx, center_y - dy),
                        (center_x + dx, center_y + dy), (0, 255, 0), 1)

    return viz


# Load a sample digit and visualize HOG
digits_path = get_sample_path("digits.png")
digits_img = cv2.imread(digits_path, cv2.IMREAD_GRAYSCALE)

if digits_img is not None:
    # Extract first few digits
    sample_digits = []
    rows = np.vsplit(digits_img, 50)
    for digit_class in range(10):
        digit = np.hsplit(rows[digit_class * 5], 100)[0]
        sample_digits.append(digit)

    # Compute HOG for first digit
    test_digit = sample_digits[3]  # Digit '3'
    hog_features = hog.compute(test_digit)
    print(f"\nHOG feature vector for digit '3': {hog_features.shape}")
else:
    print("Could not load digits.png for visualization")


# =============================================================================
# 3. HOG PARAMETERS AND EFFECTS
# =============================================================================
print("\n--- 3. HOG Parameters ---")

param_info = """
HOG Parameter Effects:

| Parameter    | Smaller Value           | Larger Value            |
|-------------|-------------------------|-------------------------|
| cellSize    | More detailed features  | Coarser features        |
| blockSize   | Less normalization      | More robust to lighting |
| nbins       | Coarse orientations     | Fine orientations       |
| blockStride | Overlapping (more feat) | Less overlap (faster)   |

Typical Settings:
  - People detection: cellSize=8, blockSize=16, nbins=9
  - Digit recognition: cellSize=5, blockSize=10, nbins=9
  - Fine details: smaller cells, more bins
  - Speed priority: larger cells, less overlap
"""
print(param_info)


# =============================================================================
# 4. HOG + SVM FOR DIGIT CLASSIFICATION
# =============================================================================
print("\n--- 4. HOG + SVM Digit Classification ---")


def load_digits_with_hog():
    """Load digits and extract HOG features."""
    digits_path = get_sample_path("digits.png")
    img = cv2.imread(digits_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Could not load digits.png")

    # Split into individual digits
    rows = np.vsplit(img, 50)
    digits = []
    for row in rows:
        digits.extend(np.hsplit(row, 100))
    digits = np.array(digits)

    # Create labels
    labels = np.repeat(np.arange(10), 500)

    # Extract HOG features for each digit
    print("Extracting HOG features...")
    hog_features = []
    for digit in digits:
        features = hog.compute(digit)
        hog_features.append(features.flatten())

    hog_features = np.array(hog_features, dtype=np.float32)
    labels = labels.astype(np.int32)

    print(f"HOG features shape: {hog_features.shape}")
    return digits, hog_features, labels


digits, hog_data, labels = load_digits_with_hog()

# Split data
np.random.seed(42)
indices = np.random.permutation(len(hog_data))
split = int(len(hog_data) * 0.8)

train_data = hog_data[indices[:split]]
train_labels = labels[indices[:split]]
test_data = hog_data[indices[split:]]
test_labels = labels[indices[split:]]

# Also prepare raw pixel data for comparison
raw_data = digits.reshape(len(digits), -1).astype(np.float32) / 255.0
raw_train = raw_data[indices[:split]]
raw_test = raw_data[indices[split:]]

print(f"\nTraining samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Train SVM with HOG features
print("\nTraining SVM with HOG features...")
svm_hog = cv2.ml.SVM_create()
svm_hog.setType(cv2.ml.SVM_C_SVC)
svm_hog.setKernel(cv2.ml.SVM_RBF)
svm_hog.setC(2.5)
svm_hog.setGamma(0.5)
svm_hog.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Evaluate
_, hog_predictions = svm_hog.predict(test_data)
hog_accuracy = np.sum(hog_predictions.flatten() == test_labels) / len(test_labels) * 100

# Train SVM with raw pixels for comparison
print("Training SVM with raw pixels...")
svm_raw = cv2.ml.SVM_create()
svm_raw.setType(cv2.ml.SVM_C_SVC)
svm_raw.setKernel(cv2.ml.SVM_RBF)
svm_raw.setC(2.5)
svm_raw.setGamma(0.5)
svm_raw.train(raw_train, cv2.ml.ROW_SAMPLE, train_labels)

_, raw_predictions = svm_raw.predict(raw_test)
raw_accuracy = np.sum(raw_predictions.flatten() == test_labels) / len(test_labels) * 100

print("\n" + "=" * 50)
print("Comparison: HOG Features vs Raw Pixels")
print("=" * 50)
print(f"Raw Pixels ({raw_train.shape[1]} features):  {raw_accuracy:.2f}%")
print(f"HOG Features ({train_data.shape[1]} features): {hog_accuracy:.2f}%")
print("=" * 50)

hog_benefit = """
Why HOG Features?
  - More robust to lighting variations
  - Captures shape/edge information
  - Works well with linear classifiers
  - Standard for pedestrian detection
  - May have fewer features than raw pixels
"""
print(hog_benefit)


# =============================================================================
# 5. PEDESTRIAN DETECTION WITH BUILT-IN HOG
# =============================================================================
print("\n--- 5. Pedestrian Detection ---")


def run_pedestrian_detection():
    """Demo of OpenCV's built-in HOG pedestrian detector."""

    # Create HOG descriptor with default people detector
    hog_detector = cv2.HOGDescriptor()
    hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Load video
    video_path = get_sample_path("vtest.avi")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video file")
        return

    print("\nRunning pedestrian detection on video...")
    print("Press 'q' to quit, 'p' to pause")

    paused = False
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1

            # Detect pedestrians
            # Returns: boxes, weights
            boxes, weights = hog_detector.detectMultiScale(
                frame,
                winStride=(8, 8),      # Detection stride
                padding=(4, 4),        # Padding
                scale=1.05,            # Scale pyramid factor
                hitThreshold=0,        # Detection threshold
                finalThreshold=2.0     # Final threshold (suppression)
            )

            # Draw detections
            for (x, y, w, h), weight in zip(boxes, weights):
                # Color based on confidence
                color = (0, int(min(255, weight * 100)), 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{weight:.2f}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Add info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Detections: {len(boxes)}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "q:quit p:pause", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("HOG Pedestrian Detection", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_hog_digits():
    """Visualize HOG features for digits."""

    # Create visualization window
    viz_height = 100
    viz_width = 10 * 80  # 10 digits, 80px each

    viz = np.zeros((viz_height, viz_width), dtype=np.uint8)

    for digit_class in range(10):
        # Get a sample digit
        digit_indices = np.where(labels == digit_class)[0]
        digit_img = digits[digit_indices[0]]

        # Compute HOG visualization
        hog_viz = visualize_hog(digit_img, hog, cell_size=5, scale=3)

        # Place original and HOG side by side
        x_offset = digit_class * 80

        # Original (scaled)
        original = cv2.resize(digit_img, (40, 40), interpolation=cv2.INTER_NEAREST)
        viz[10:50, x_offset:x_offset+40] = original

        # HOG visualization (convert to grayscale)
        hog_gray = cv2.cvtColor(hog_viz, cv2.COLOR_BGR2GRAY)
        hog_resized = cv2.resize(hog_gray, (40, 40), interpolation=cv2.INTER_AREA)
        viz[10:50, x_offset+40:x_offset+80] = hog_resized

        # Label
        cv2.putText(viz, str(digit_class), (x_offset + 35, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    # Scale up for visibility
    viz_large = cv2.resize(viz, (viz_width * 2, viz_height * 2), interpolation=cv2.INTER_NEAREST)

    cv2.putText(viz_large, "Original | HOG (each pair)", (10, 180),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

    cv2.imshow("HOG Features for Digits 0-9", viz_large)

    # Show misclassified examples
    hog_pred = hog_predictions.flatten()
    misclassified = np.where(hog_pred != test_labels)[0]

    print(f"\nMisclassified: {len(misclassified)} / {len(test_labels)}")

    if len(misclassified) > 0:
        n_show = min(10, len(misclassified))
        mis_viz = np.zeros((80, n_show * 50), dtype=np.uint8)

        for i in range(n_show):
            idx = misclassified[i]
            global_idx = indices[split:][idx]  # Get original index
            digit_img = digits[global_idx]

            # Show digit
            digit_scaled = cv2.resize(digit_img, (40, 40), interpolation=cv2.INTER_NEAREST)
            mis_viz[5:45, i*50+5:i*50+45] = digit_scaled

            # Labels
            true_label = test_labels[idx]
            pred_label = int(hog_pred[idx])
            cv2.putText(mis_viz, f"T:{true_label} P:{pred_label}", (i*50+2, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, 200, 1)

        cv2.imshow("HOG+SVM Misclassified", mis_viz)

    print("\nPress any key to continue to pedestrian detection...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Visualizing HOG Features...")
    print("=" * 60)

    visualize_hog_digits()

    print("\n" + "=" * 60)
    print("Starting Pedestrian Detection Demo...")
    print("=" * 60)

    run_pedestrian_detection()
