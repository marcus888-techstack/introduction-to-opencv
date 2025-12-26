"""
Module 9: Machine Learning - Decision Trees
============================================
Decision Trees and model comparison for digit classification.

Official Docs: https://docs.opencv.org/4.x/d0/d72/tutorial_py_dtree.html

Topics Covered:
1. Decision Trees for Digit Recognition
2. Tree Parameters and Overfitting
3. Cross-Validation Concepts
4. Model Saving and Loading
5. Comprehensive Algorithm Comparison

Real Data Used:
- digits.png: 5000 handwritten digits (0-9), each 20x20 pixels
"""

import cv2
import numpy as np
import os
import sys
import time

# Add parent directory to path for sample_data imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data.download_samples import get_sample_path

print("=" * 60)
print("Module 9: Decision Trees & Model Comparison")
print("=" * 60)


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
def load_and_prepare_digits():
    """Load digits.png and prepare for ML."""
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

    # Flatten and normalize
    data = digits.reshape(len(digits), -1).astype(np.float32) / 255.0
    labels = labels.astype(np.int32)

    # Split into train/test (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    split = int(len(data) * 0.8)

    train_data = data[indices[:split]]
    train_labels = labels[indices[:split]]
    test_data = data[indices[split:]]
    test_labels = labels[indices[split:]]

    return train_data, train_labels, test_data, test_labels, digits, labels


print("\n--- Loading Data ---")
train_data, train_labels, test_data, test_labels, all_digits, all_labels = load_and_prepare_digits()
print(f"Training: {len(train_data)}, Test: {len(test_data)}")


# =============================================================================
# 1. DECISION TREES FOR DIGIT RECOGNITION
# =============================================================================
print("\n--- 1. Decision Trees ---")

# Create Decision Tree classifier
dtree = cv2.ml.DTrees_create()

# Set parameters
dtree.setMaxDepth(10)           # Maximum tree depth
dtree.setMinSampleCount(5)      # Minimum samples to split a node
dtree.setCVFolds(0)             # No cross-validation during training
dtree.setMaxCategories(10)      # Maximum number of categories

# Train
start_time = time.time()
dtree.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
train_time = time.time() - start_time

# Predict
start_time = time.time()
_, dtree_results = dtree.predict(test_data)
predict_time = time.time() - start_time

# Calculate accuracy
dtree_accuracy = np.sum(dtree_results.flatten() == test_labels) / len(test_labels) * 100

print(f"Decision Tree Accuracy: {dtree_accuracy:.2f}%")
print(f"Training time: {train_time:.3f}s")
print(f"Prediction time: {predict_time:.3f}s")

dtree_info = """
Decision Tree Characteristics:
  - Creates hierarchical rules based on feature thresholds
  - Easy to interpret (can visualize the tree)
  - Fast training and prediction
  - Prone to overfitting without depth limits
  - Works well with categorical features
"""
print(dtree_info)


# =============================================================================
# 2. TREE PARAMETERS AND OVERFITTING
# =============================================================================
print("\n--- 2. Effect of Tree Depth on Overfitting ---")

print("\nDepth | Train Acc | Test Acc | Diff (Overfitting)")
print("-" * 55)

for max_depth in [2, 5, 8, 10, 12, 15]:
    tree = cv2.ml.DTrees_create()
    tree.setMaxDepth(max_depth)
    tree.setMinSampleCount(5)  # Prevent overly deep trees

    tree.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    # Training accuracy
    _, train_pred = tree.predict(train_data)
    train_acc = np.sum(train_pred.flatten() == train_labels) / len(train_labels) * 100

    # Test accuracy
    _, test_pred = tree.predict(test_data)
    test_acc = np.sum(test_pred.flatten() == test_labels) / len(test_labels) * 100

    diff = train_acc - test_acc
    print(f"{max_depth:5} | {train_acc:8.2f}% | {test_acc:7.2f}% | {diff:+.2f}%")

overfitting_info = """
Overfitting Analysis:
  - Large (Train - Test) gap indicates overfitting
  - Deeper trees memorize training data
  - Shallower trees generalize better
  - Optimal depth balances accuracy and generalization
"""
print(overfitting_info)


# =============================================================================
# 3. CROSS-VALIDATION CONCEPTS
# =============================================================================
print("\n--- 3. K-Fold Cross-Validation ---")


def cross_validate(data, labels, n_folds=5, create_model_fn=None):
    """
    Perform k-fold cross-validation.

    Args:
        data: Feature matrix
        labels: Label vector
        n_folds: Number of folds
        create_model_fn: Function that creates and configures a model

    Returns:
        List of accuracies for each fold
    """
    n_samples = len(data)
    fold_size = n_samples // n_folds

    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    accuracies = []

    for fold in range(n_folds):
        # Create train/validation split
        val_start = fold * fold_size
        val_end = val_start + fold_size

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        fold_train_data = data[train_idx]
        fold_train_labels = labels[train_idx]
        fold_val_data = data[val_idx]
        fold_val_labels = labels[val_idx]

        # Create and train model
        model = create_model_fn()
        model.train(fold_train_data, cv2.ml.ROW_SAMPLE, fold_train_labels)

        # Evaluate
        _, predictions = model.predict(fold_val_data)
        accuracy = np.sum(predictions.flatten() == fold_val_labels) / len(fold_val_labels) * 100
        accuracies.append(accuracy)

    return accuracies


# Cross-validate Decision Tree
def create_dtree():
    tree = cv2.ml.DTrees_create()
    tree.setMaxDepth(10)
    tree.setMinSampleCount(5)
    return tree


# Combine train and test for cross-validation
all_data = np.vstack([train_data, test_data])
all_labels_cv = np.concatenate([train_labels, test_labels])

print("5-Fold Cross-Validation Results:")
print("-" * 40)

dtree_cv_accs = cross_validate(all_data, all_labels_cv, n_folds=5, create_model_fn=create_dtree)
for i, acc in enumerate(dtree_cv_accs):
    print(f"  Fold {i+1}: {acc:.2f}%")
print(f"  Mean: {np.mean(dtree_cv_accs):.2f}% (+/- {np.std(dtree_cv_accs):.2f}%)")

cv_info = """
Cross-Validation Benefits:
  - Uses all data for both training and validation
  - Reduces variance in accuracy estimates
  - Helps detect overfitting
  - Standard for comparing algorithms
"""
print(cv_info)


# =============================================================================
# 4. MODEL SAVING AND LOADING
# =============================================================================
print("\n--- 4. Model Persistence ---")

# Create a temporary directory for models
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)

# Save Decision Tree
dtree_path = os.path.join(model_dir, 'dtree_digits.xml')
dtree.save(dtree_path)
print(f"Saved Decision Tree to: {dtree_path}")

# Load it back
loaded_dtree = cv2.ml.DTrees_load(dtree_path)
_, loaded_pred = loaded_dtree.predict(test_data)
loaded_acc = np.sum(loaded_pred.flatten() == test_labels) / len(test_labels) * 100
print(f"Loaded model accuracy: {loaded_acc:.2f}%")

# Train and save SVM for comparison
print("\nTraining SVM for comparison...")
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(2.5)
svm.setGamma(0.5)
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

svm_path = os.path.join(model_dir, 'svm_digits.xml')
svm.save(svm_path)
print(f"Saved SVM to: {svm_path}")

# Train and save KNN
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
# Note: KNN can't be saved the same way (it stores all training data)
print("Note: KNN stores all training data, not model parameters")

persistence_info = """
Model Persistence Functions:
  model.save('path.xml')     - Save model to XML file
  cv2.ml.SVM_load('path')    - Load SVM model
  cv2.ml.DTrees_load('path') - Load Decision Tree
  cv2.ml.KNearest_load('path') - Load KNN (limited)
"""
print(persistence_info)


# =============================================================================
# 5. COMPREHENSIVE ALGORITHM COMPARISON
# =============================================================================
print("\n--- 5. Algorithm Comparison ---")


def evaluate_model(model, train_data, train_labels, test_data, test_labels, name, is_knn=False):
    """Evaluate a model and return metrics."""
    start = time.time()
    model.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    train_time = time.time() - start

    start = time.time()
    if is_knn:
        _, results, _, _ = model.findNearest(test_data, k=3)
    else:
        _, results = model.predict(test_data)
    predict_time = time.time() - start

    accuracy = np.sum(results.flatten() == test_labels) / len(test_labels) * 100

    return {
        'name': name,
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time
    }


# Compare all algorithms
results = []

# KNN
knn = cv2.ml.KNearest_create()
results.append(evaluate_model(knn, train_data, train_labels, test_data, test_labels, 'KNN (k=3)', is_knn=True))

# SVM with different kernels
for kernel_name, kernel_type in [('Linear', cv2.ml.SVM_LINEAR), ('RBF', cv2.ml.SVM_RBF)]:
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(kernel_type)
    svm.setC(2.5)
    if kernel_type == cv2.ml.SVM_RBF:
        svm.setGamma(0.5)
    results.append(evaluate_model(svm, train_data, train_labels, test_data, test_labels, f'SVM ({kernel_name})'))

# Decision Tree with different depths
for depth in [5, 10, 15]:
    dtree = cv2.ml.DTrees_create()
    dtree.setMaxDepth(depth)
    dtree.setMinSampleCount(5)
    results.append(evaluate_model(dtree, train_data, train_labels, test_data, test_labels, f'DTree (d={depth})'))

# Print comparison table
print("\n" + "=" * 70)
print(f"{'Algorithm':<20} {'Accuracy':>10} {'Train Time':>12} {'Predict Time':>12}")
print("=" * 70)

for r in sorted(results, key=lambda x: -x['accuracy']):
    print(f"{r['name']:<20} {r['accuracy']:>9.2f}% {r['train_time']:>11.3f}s {r['predict_time']:>11.3f}s")

print("=" * 70)


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_confusion_matrix():
    """Show confusion matrix for best model."""
    # Use SVM RBF as best model
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2.5)
    svm.setGamma(0.5)
    svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    _, predictions = svm.predict(test_data)
    predictions = predictions.flatten().astype(int)

    # Build confusion matrix
    confusion = np.zeros((10, 10), dtype=int)
    for true, pred in zip(test_labels, predictions):
        confusion[true, pred] += 1

    # Visualize
    cell_size = 40
    viz = np.zeros((10 * cell_size + 50, 10 * cell_size + 50), dtype=np.uint8)
    viz[:] = 255

    for i in range(10):
        for j in range(10):
            count = confusion[i, j]
            # Color intensity based on count
            intensity = min(255, count * 5)
            color = 255 - intensity

            x1 = j * cell_size + 50
            y1 = i * cell_size + 50
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            cv2.rectangle(viz, (x1, y1), (x2, y2), int(color), -1)
            cv2.rectangle(viz, (x1, y1), (x2, y2), 0, 1)

            # Add count text
            text = str(count)
            text_color = 255 if intensity > 127 else 0
            cv2.putText(viz, text, (x1 + 10, y1 + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    # Add labels
    for i in range(10):
        # Row labels (true)
        cv2.putText(viz, str(i), (25, i * cell_size + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        # Column labels (predicted)
        cv2.putText(viz, str(i), (i * cell_size + 65, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    cv2.putText(viz, "True", (5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    cv2.putText(viz, "Predicted", (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

    cv2.imshow("Confusion Matrix (SVM RBF)", viz)

    # Show misclassified examples
    misclassified = np.where(predictions != test_labels)[0]
    if len(misclassified) > 0:
        n_show = min(10, len(misclassified))
        mis_viz = np.zeros((60, n_show * 40), dtype=np.uint8)

        for i in range(n_show):
            idx = misclassified[i]
            digit_img = (test_data[idx] * 255).astype(np.uint8).reshape(20, 20)
            digit_img = cv2.resize(digit_img, (40, 40), interpolation=cv2.INTER_NEAREST)

            mis_viz[0:40, i*40:(i+1)*40] = digit_img

            true_label = test_labels[idx]
            pred_label = predictions[idx]
            cv2.putText(mis_viz, f"{true_label}->{pred_label}", (i*40 + 2, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, 200, 1)

        cv2.imshow("Misclassified Examples (True->Pred)", mis_viz)

    print("\nConfusion Matrix Analysis:")
    print("-" * 40)
    print("Common confusions:")
    for i in range(10):
        for j in range(10):
            if i != j and confusion[i, j] >= 3:
                print(f"  {i} misclassified as {j}: {confusion[i, j]} times")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Visualizing Results...")
    print("=" * 60)
    visualize_confusion_matrix()
