"""
Module 9: Machine Learning - Basics
===================================
Traditional machine learning with OpenCV.

Official Docs: https://docs.opencv.org/4.x/d6/de2/tutorial_py_table_of_contents_ml.html

Topics Covered:
1. K-Nearest Neighbors (KNN)
2. Support Vector Machines (SVM)
3. K-Means Clustering
4. Decision Trees
5. Data Preparation
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 9: Machine Learning")
print("=" * 60)


# =============================================================================
# 1. GENERATING SAMPLE DATA
# =============================================================================
print("\n--- 1. Generating Sample Data ---")


def generate_classification_data(n_samples=100):
    """Generate 2D data for classification demo."""
    np.random.seed(42)

    # Class 0: Cluster around (50, 50)
    class0_x = np.random.normal(50, 15, n_samples)
    class0_y = np.random.normal(50, 15, n_samples)

    # Class 1: Cluster around (150, 150)
    class1_x = np.random.normal(150, 15, n_samples)
    class1_y = np.random.normal(150, 15, n_samples)

    # Combine
    X = np.vstack([
        np.column_stack([class0_x, class0_y]),
        np.column_stack([class1_x, class1_y])
    ]).astype(np.float32)

    y = np.array([0]*n_samples + [1]*n_samples, dtype=np.int32)

    return X, y


X_train, y_train = generate_classification_data(80)
X_test, y_test = generate_classification_data(20)

print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test data: {X_test.shape}, Labels: {y_test.shape}")


# =============================================================================
# 2. K-NEAREST NEIGHBORS (KNN)
# =============================================================================
print("\n--- 2. K-Nearest Neighbors ---")

# Create KNN classifier
knn = cv2.ml.KNearest_create()

# Train
knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Predict
k = 5  # Number of neighbors
ret, results, neighbors, dist = knn.findNearest(X_test, k)

# Calculate accuracy
accuracy = np.sum(results.flatten() == y_test) / len(y_test) * 100
print(f"KNN (k={k}) Accuracy: {accuracy:.1f}%")

knn_info = """
KNN Parameters:
  k              - Number of neighbors
  algorithmType  - BRUTE_FORCE or KDTREE

Methods:
  train(samples, layout, responses) - Train classifier
  findNearest(samples, k) - Find k nearest neighbors

Returns:
  ret       - Return value
  results   - Predicted labels
  neighbors - Labels of k neighbors
  dist      - Distances to k neighbors
"""
print(knn_info)


# =============================================================================
# 3. SUPPORT VECTOR MACHINES (SVM)
# =============================================================================
print("\n--- 3. Support Vector Machines ---")

# Create SVM classifier
svm = cv2.ml.SVM_create()

# Set parameters
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(1.0)
svm.setGamma(0.5)

# Train
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Predict
_, svm_results = svm.predict(X_test)

# Calculate accuracy
svm_accuracy = np.sum(svm_results.flatten() == y_test) / len(y_test) * 100
print(f"SVM Accuracy: {svm_accuracy:.1f}%")

svm_info = """
SVM Types:
  SVM_C_SVC     - C-Support Vector Classification
  SVM_NU_SVC    - Nu-Support Vector Classification
  SVM_ONE_CLASS - One-class SVM (anomaly detection)
  SVM_EPS_SVR   - Epsilon-Support Vector Regression
  SVM_NU_SVR    - Nu-Support Vector Regression

Kernels:
  SVM_LINEAR    - Linear: u'*v
  SVM_POLY      - Polynomial: (gamma*u'*v + coef0)^degree
  SVM_RBF       - RBF: exp(-gamma*|u-v|^2)
  SVM_SIGMOID   - Sigmoid: tanh(gamma*u'*v + coef0)

Parameters:
  C      - Penalty parameter (higher = less tolerance for misclassification)
  Gamma  - Kernel coefficient
"""
print(svm_info)


# =============================================================================
# 4. K-MEANS CLUSTERING
# =============================================================================
print("\n--- 4. K-Means Clustering ---")

# Generate clustering data
np.random.seed(42)
cluster1 = np.random.normal(50, 10, (50, 2))
cluster2 = np.random.normal(150, 15, (50, 2))
cluster3 = np.random.normal(100, 20, (50, 2))
cluster3[:, 1] += 50  # Shift y

cluster_data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)

# Define criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Apply K-Means
k = 3
compactness, labels, centers = cv2.kmeans(
    cluster_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

print(f"Number of clusters: {k}")
print(f"Cluster centers:\n{centers}")
print(f"Compactness (lower is better): {compactness:.2f}")

kmeans_info = """
K-Means Parameters:
  data      - Input data (float32)
  K         - Number of clusters
  bestLabels - Output labels (can be None)
  criteria  - Termination criteria
  attempts  - Number of attempts with different initializations
  flags     - Initialization method

Flags:
  KMEANS_RANDOM_CENTERS  - Random initialization
  KMEANS_PP_CENTERS      - K-means++ initialization
  KMEANS_USE_INITIAL_LABELS - Use provided labels

Returns:
  compactness - Sum of squared distances to centers
  labels      - Cluster assignments
  centers     - Cluster centers
"""
print(kmeans_info)


# =============================================================================
# 5. DECISION TREES
# =============================================================================
print("\n--- 5. Decision Trees ---")

# Create Decision Tree classifier
dtree = cv2.ml.DTrees_create()

# Set parameters
dtree.setMaxDepth(5)
dtree.setMinSampleCount(5)

# Train
dtree.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Predict
_, dtree_results = dtree.predict(X_test)

# Calculate accuracy
dtree_accuracy = np.sum(dtree_results.flatten() == y_test) / len(y_test) * 100
print(f"Decision Tree Accuracy: {dtree_accuracy:.1f}%")

dtree_info = """
Decision Tree Parameters:
  MaxDepth        - Maximum tree depth
  MinSampleCount  - Minimum samples for splitting
  MaxCategories   - Max categories for clustering
  CVFolds         - Cross-validation folds
  Use1SERule      - Use 1-standard-error rule
  TruncatePrunedTree - Prune after training
"""
print(dtree_info)


# =============================================================================
# 6. MODEL PERSISTENCE
# =============================================================================
print("\n--- 6. Model Persistence ---")

persistence_info = """
Saving and Loading Models:

# Save model
svm.save('svm_model.xml')

# Load model
loaded_svm = cv2.ml.SVM_load('svm_model.xml')

# Alternatively, use FileStorage:
fs = cv2.FileStorage('model.yml', cv2.FILE_STORAGE_WRITE)
svm.write(fs)
fs.release()

# Load from FileStorage:
fs = cv2.FileStorage('model.yml', cv2.FILE_STORAGE_READ)
loaded_svm = cv2.ml.SVM_create()
loaded_svm.read(fs.getNode('opencv_ml_svm'))
fs.release()
"""
print(persistence_info)


# =============================================================================
# 7. ALGORITHM COMPARISON
# =============================================================================
print("\n--- 7. Algorithm Comparison ---")

comparison = """
ML Algorithm Comparison:

| Algorithm     | Type           | Speed   | Use Case                    |
|---------------|----------------|---------|------------------------------|
| KNN           | Classification | Fast    | Simple, small datasets       |
| SVM           | Classification | Medium  | High-dimensional, accurate   |
| Decision Tree | Classification | Fast    | Interpretable, feature importance|
| K-Means       | Clustering     | Fast    | Unsupervised grouping        |
| Random Forest | Classification | Medium  | Ensemble, robust             |
| Boost         | Classification | Medium  | Combine weak classifiers     |

Tips:
- Start with KNN for quick prototyping
- Use SVM for binary classification
- Decision Trees for interpretability
- K-Means for finding natural groups
"""
print(comparison)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display ML demos."""

    # Create visualization image
    viz = np.zeros((200, 200, 3), dtype=np.uint8)
    viz[:] = (255, 255, 255)

    # Draw training data
    for i, (x, y) in enumerate(X_train):
        color = (255, 0, 0) if y_train[i] == 0 else (0, 0, 255)
        cv2.circle(viz, (int(x), int(y)), 3, color, -1)

    # Draw decision boundary for SVM
    for i in range(200):
        for j in range(200):
            sample = np.array([[i, j]], dtype=np.float32)
            _, response = svm.predict(sample)
            if response[0] == 0:
                viz[j, i] = tuple(int(c * 0.9) for c in viz[j, i])

    cv2.putText(viz, "SVM Decision Boundary", (5, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(viz, "Blue: Class 0, Red: Class 1", (5, 190),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    cv2.imshow("Classification Demo", viz)

    # K-Means visualization
    kmeans_viz = np.zeros((200, 200, 3), dtype=np.uint8)
    kmeans_viz[:] = (255, 255, 255)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, (x, y) in enumerate(cluster_data):
        label = int(labels[i])
        cv2.circle(kmeans_viz, (int(x), int(y)), 3, colors[label], -1)

    # Draw centers
    for i, center in enumerate(centers):
        cv2.circle(kmeans_viz, (int(center[0]), int(center[1])), 8, colors[i], 2)
        cv2.circle(kmeans_viz, (int(center[0]), int(center[1])), 2, (0, 0, 0), -1)

    cv2.putText(kmeans_viz, "K-Means Clustering", (5, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.imshow("K-Means Demo", kmeans_viz)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running ML demonstrations...")
    print("=" * 60)
    show_demo()
