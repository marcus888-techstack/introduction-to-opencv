"""
Module 9: Machine Learning - Basics
===================================
Traditional machine learning with OpenCV using real handwritten digit images.

Official Docs: https://docs.opencv.org/4.x/d6/de2/tutorial_py_table_of_contents_ml.html

Topics Covered:
1. Loading Real Image Data (digits.png)
2. Data Preparation for ML
3. K-Nearest Neighbors (KNN) for Digit Recognition
4. Support Vector Machines (SVM) for Digit Recognition
5. Model Evaluation and Comparison

Real Data Used:
- digits.png: 5000 handwritten digits (0-9), each 20x20 pixels
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data.download_samples import get_sample_path

print("=" * 60)
print("Module 9: Machine Learning with Real Digit Images")
print("=" * 60)


# =============================================================================
# 1. LOADING REAL IMAGE DATA (digits.png)
# =============================================================================
print("\n--- 1. Loading Real Handwritten Digits ---")


def load_digits():
    """
    Load OpenCV's digits.png dataset.

    digits.png contains:
    - 5000 handwritten digits (0-9)
    - Each digit is 20x20 pixels
    - Arranged in 50 rows x 100 columns
    - 500 samples per digit class

    Returns:
        digits: All digit images as (5000, 20, 20) array
        labels: Corresponding labels (0-9)
    """
    # Download and load digits.png
    digits_path = get_sample_path("digits.png")
    img = cv2.imread(digits_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load digits.png from {digits_path}")

    print(f"Loaded digits.png: {img.shape}")

    # Split into 5000 individual digits (50 rows x 100 cols = 5000 digits)
    # Each digit is 20x20 pixels
    # Total image is 1000x2000 (50*20 x 100*20)

    rows = np.vsplit(img, 50)  # Split into 50 rows
    digits = []
    for row in rows:
        row_digits = np.hsplit(row, 100)  # Split each row into 100 digits
        digits.extend(row_digits)

    digits = np.array(digits)  # Shape: (5000, 20, 20)

    # Create labels: 0-9, each repeated 500 times
    labels = np.repeat(np.arange(10), 500)  # [0,0,...,0,1,1,...,9,9,9]

    print(f"Extracted {len(digits)} digits, each {digits[0].shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Samples per class: {np.bincount(labels)}")

    return digits, labels


digits, labels = load_digits()


# =============================================================================
# 2. DATA PREPARATION FOR ML
# =============================================================================
print("\n--- 2. Data Preparation ---")


def prepare_data(digits, labels, train_ratio=0.8):
    """
    Prepare data for machine learning.

    Steps:
    1. Flatten 20x20 images to 400-element vectors
    2. Normalize pixel values to [0, 1]
    3. Split into training and test sets

    Args:
        digits: (N, 20, 20) array of digit images
        labels: (N,) array of labels
        train_ratio: Fraction for training

    Returns:
        train_data, train_labels, test_data, test_labels
    """
    # Flatten: (5000, 20, 20) -> (5000, 400)
    n_samples = len(digits)
    data = digits.reshape(n_samples, -1).astype(np.float32)

    # Normalize to [0, 1] - important for SVM
    data = data / 255.0

    # Labels must be int32 for OpenCV ML
    labels = labels.astype(np.int32)

    # Split into train/test
    split_idx = int(n_samples * train_ratio)

    # Shuffle indices for random split
    np.random.seed(42)
    indices = np.random.permutation(n_samples)

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_data = data[train_idx]
    train_labels = labels[train_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]

    print(f"Feature vector size: {data.shape[1]} (flattened 20x20)")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    return train_data, train_labels, test_data, test_labels


train_data, train_labels, test_data, test_labels = prepare_data(digits, labels)


# =============================================================================
# 3. K-NEAREST NEIGHBORS (KNN) FOR DIGIT RECOGNITION
# =============================================================================
print("\n--- 3. KNN Digit Recognition ---")

# Create KNN classifier
knn = cv2.ml.KNearest_create()

# Train with training data
# ROW_SAMPLE means each row is a sample (standard format)
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Test with different k values
print("\nKNN Results for different k values:")
print("-" * 40)

for k in [1, 3, 5, 7]:
    # findNearest returns: retval, results, neighbours, dist
    ret, results, neighbors, dist = knn.findNearest(test_data, k)

    # Calculate accuracy
    matches = (results.flatten() == test_labels)
    accuracy = np.sum(matches) / len(test_labels) * 100

    print(f"k={k}: Accuracy = {accuracy:.2f}%")

# Use k=3 as default
k = 3
ret, results, neighbors, dist = knn.findNearest(test_data, k)
knn_accuracy = np.sum(results.flatten() == test_labels) / len(test_labels) * 100

knn_info = """
KNN for Digit Recognition:
  - Compares test digit to all training digits
  - Uses Euclidean distance on 400-dim feature vectors
  - Majority vote among k nearest neighbors
  - No training phase (lazy learning)
  - Fast for small datasets, slow for large ones
"""
print(knn_info)


# =============================================================================
# 4. SUPPORT VECTOR MACHINE (SVM) FOR DIGIT RECOGNITION
# =============================================================================
print("\n--- 4. SVM Digit Recognition ---")

# Create SVM classifier
svm = cv2.ml.SVM_create()

# Set parameters for multi-class classification
svm.setType(cv2.ml.SVM_C_SVC)      # C-Support Vector Classification
svm.setKernel(cv2.ml.SVM_RBF)       # Radial Basis Function kernel
svm.setC(2.5)                        # Penalty parameter
svm.setGamma(0.5)                    # Kernel coefficient

# Train SVM (takes longer than KNN)
print("Training SVM (this may take a moment)...")
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
print("Training complete!")

# Predict on test data
_, svm_results = svm.predict(test_data)

# Calculate accuracy
svm_accuracy = np.sum(svm_results.flatten() == test_labels) / len(test_labels) * 100
print(f"SVM Accuracy: {svm_accuracy:.2f}%")

svm_info = """
SVM for Digit Recognition:
  - Finds optimal hyperplane to separate classes
  - RBF kernel maps to high-dimensional space
  - Handles non-linear boundaries well
  - Slower training, fast prediction
  - Often more accurate than KNN
"""
print(svm_info)


# =============================================================================
# 5. MODEL COMPARISON
# =============================================================================
print("\n--- 5. Model Comparison ---")

comparison = f"""
+------------------+------------+---------------+
|   Algorithm      | Accuracy   | Characteristics|
+------------------+------------+---------------+
| KNN (k=3)        | {knn_accuracy:6.2f}%   | Fast training |
| SVM (RBF)        | {svm_accuracy:6.2f}%   | More accurate |
+------------------+------------+---------------+

When to use which:
  - KNN: Quick prototyping, interpretable results
  - SVM: Higher accuracy needed, multi-class problems
"""
print(comparison)


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_digits_and_predictions():
    """Show sample digits with predictions."""

    # Show sample digits from dataset
    viz_samples = np.zeros((100, 200), dtype=np.uint8)

    # Show 2 samples of each digit (0-9)
    for digit in range(10):
        # Find samples of this digit
        digit_indices = np.where(labels == digit)[0]

        for i in range(2):
            sample = digits[digit_indices[i]]
            row = i
            col = digit
            viz_samples[row*20:(row+1)*20, col*20:(col+1)*20] = sample

    # Scale up for visibility
    viz_samples = cv2.resize(viz_samples, (400, 200), interpolation=cv2.INTER_NEAREST)

    # Add labels
    for digit in range(10):
        cv2.putText(viz_samples, str(digit), (digit*40 + 15, 195),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    cv2.imshow("Sample Digits (0-9)", viz_samples)

    # Show some predictions with confidence
    n_show = 10
    test_indices = np.random.choice(len(test_data), n_show, replace=False)

    pred_viz = np.zeros((60, n_show * 40), dtype=np.uint8)

    for i, idx in enumerate(test_indices):
        # Get the digit image (reshape from flattened)
        digit_img = (test_data[idx] * 255).astype(np.uint8).reshape(20, 20)
        digit_img = cv2.resize(digit_img, (40, 40), interpolation=cv2.INTER_NEAREST)

        # Place in visualization
        pred_viz[0:40, i*40:(i+1)*40] = digit_img

        # Get prediction
        true_label = test_labels[idx]
        _, pred = svm.predict(test_data[idx:idx+1])
        pred_label = int(pred[0])

        # Color: green for correct, red for wrong
        color = 255 if pred_label == true_label else 128

        # Add prediction text
        cv2.putText(pred_viz, f"{pred_label}", (i*40 + 15, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imshow("SVM Predictions (predicted labels below)", pred_viz)

    # Show confusion analysis
    print("\nPer-class accuracy (SVM):")
    print("-" * 30)
    for digit in range(10):
        digit_mask = (test_labels == digit)
        digit_predictions = svm_results.flatten()[digit_mask]
        digit_accuracy = np.sum(digit_predictions == digit) / np.sum(digit_mask) * 100
        print(f"Digit {digit}: {digit_accuracy:.1f}%")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interactive_demo():
    """Draw your own digit and classify it."""

    canvas = np.zeros((200, 200), dtype=np.uint8)
    drawing = False

    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(canvas, (x, y), 8, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw a Digit (0-9)")
    cv2.setMouseCallback("Draw a Digit (0-9)", draw)

    print("\n" + "=" * 50)
    print("Interactive Digit Classification")
    print("=" * 50)
    print("Draw a digit with your mouse")
    print("Press 'c' to classify")
    print("Press 'r' to clear")
    print("Press 'q' to quit")

    while True:
        # Add instructions to display
        display = canvas.copy()
        cv2.putText(display, "c:classify r:clear q:quit", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 128, 1)

        cv2.imshow("Draw a Digit (0-9)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Classify
            # Resize to 20x20
            digit = cv2.resize(canvas, (20, 20), interpolation=cv2.INTER_AREA)

            # Prepare for prediction
            sample = digit.reshape(1, -1).astype(np.float32) / 255.0

            # KNN prediction
            _, knn_pred, _, knn_dist = knn.findNearest(sample, k=3)

            # SVM prediction
            _, svm_pred = svm.predict(sample)

            print(f"\nKNN prediction: {int(knn_pred[0][0])}")
            print(f"SVM prediction: {int(svm_pred[0][0])}")

            # Show the processed digit
            processed = cv2.resize(digit, (100, 100), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Processed (20x20)", processed)

        elif key == ord('r'):  # Reset
            canvas = np.zeros((200, 200), dtype=np.uint8)

        elif key == ord('q'):  # Quit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running ML demonstrations with real digits...")
    print("=" * 60)

    # Show digit samples and predictions
    visualize_digits_and_predictions()

    # Interactive drawing demo
    print("\nStarting interactive demo...")
    interactive_demo()
