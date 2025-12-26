# Module 9: Machine Learning

Traditional machine learning algorithms in OpenCV for classification, clustering, and image analysis using **real images**.

## Topics Covered

- K-Nearest Neighbors (KNN) for digit recognition
- Support Vector Machines (SVM) for classification
- Decision Trees and overfitting analysis
- HOG (Histogram of Oriented Gradients) features
- K-Means clustering for image segmentation
- Model persistence (save/load)
- Cross-validation concepts

---

## Real Data Used

| File | Description | Use Case |
|------|-------------|----------|
| `digits.png` | 5000 handwritten digits (20x20 each) | KNN, SVM, Decision Tree classification |
| `fruits.jpg` | Colorful fruit image | K-Means color segmentation |
| `vtest.avi` | Pedestrian video | HOG person detection |

---

## Tutorial Files

| File | Topics | Key Functions |
|------|--------|---------------|
| `01_ml_basics.py` | KNN, SVM with real digits | `KNearest_create()`, `SVM_create()` |
| `02_digit_recognition.py` | Decision Trees, cross-validation | `DTrees_create()`, model persistence |
| `03_hog_svm.py` | HOG features, pedestrian detection | `HOGDescriptor()`, `detectMultiScale()` |
| `04_kmeans_segmentation.py` | Color quantization, segmentation | `cv2.kmeans()`, elbow method |

---

## Algorithm Explanations

### 1. K-Nearest Neighbors (KNN)

**What it does**: Classifies samples based on the majority vote of K nearest training samples.

**KNN Classification Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    K-Nearest Neighbors (K=3)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│         Class A (●)         Class B (○)                            │
│                                                                     │
│              ●                                                      │
│                    ○                        ○                       │
│         ●               ╭─────╮                                    │
│                        ╱   ●   ╲      New point: ★                │
│              ○       ╱    ★    ╲                                   │
│                     │      ○    │     Find 3 nearest neighbors     │
│         ●          │     ●      │     • 2 are ●(Class A)           │
│                      ╲         ╱      • 1 is ○(Class B)            │
│                        ╲     ╱        Majority vote → Class A      │
│              ○           ╰───╯                                      │
│                                   ○                                 │
│         ●                                                           │
│                                                                     │
│   No training phase! Just store training data and compute at       │
│   prediction time (lazy learning)                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Distance Metric** (Euclidean):
```
d(x, y) = √Σᵢ(xᵢ - yᵢ)²
```

**OpenCV**:
```python
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)

ret, results, neighbors, dist = knn.findNearest(test_data, k=5)
```

---

### 2. Support Vector Machines (SVM)

**What it does**: Finds the optimal hyperplane that separates classes with maximum margin.

**SVM Maximum Margin Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    SVM: Maximum Margin Classifier                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Bad Boundary           Better Boundary        Optimal (SVM)      │
│                                                                     │
│   ●  ●  │  ○  ○          ●  ●  │  ○  ○         ●  ●  ┃  ○  ○      │
│   ●  ●  │  ○  ○          ●  ●  │  ○  ○         ●  ●  ┃  ○  ○      │
│   ●  ●  │  ○  ○          ●  ●   │  ○  ○        ●  ●  ┃  ○  ○      │
│                         ◀──margin──▶                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Kernel Functions**:
| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | K(u,v) = uᵀv | Linearly separable |
| RBF | K(u,v) = exp(-γ‖u-v‖²) | General purpose (default) |
| Polynomial | K(u,v) = (γuᵀv + r)^d | Polynomial boundaries |

**OpenCV**:
```python
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(2.5)        # Regularization
svm.setGamma(0.5)    # Kernel coefficient

svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)
_, prediction = svm.predict(test_data)
```

---

### 3. Decision Trees

**What it does**: Recursively splits data based on feature thresholds.

**Decision Tree Structure**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Decision Tree Visualization                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                     ┌───────────────┐                               │
│                     │  Pixel > 128? │ ← Root node                  │
│                     └───────┬───────┘                               │
│                  ┌──────────┴──────────┐                            │
│              Yes │                     │ No                         │
│                  ▼                     ▼                            │
│         ┌───────────────┐     ┌───────────────┐                    │
│         │   Feature A?  │     │   Feature B?  │                    │
│         └───────┬───────┘     └───────┬───────┘                    │
│              ┌──┴──┐               ┌──┴──┐                          │
│              ▼     ▼               ▼     ▼                          │
│           [Digit 3] [Digit 8]  [Digit 1] [Digit 7]                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Overfitting Control**:
| Parameter | Effect |
|-----------|--------|
| `MaxDepth` | Limits tree depth (prevents overfitting) |
| `MinSampleCount` | Minimum samples to split a node |

**OpenCV**:
```python
dtree = cv2.ml.DTrees_create()
dtree.setMaxDepth(10)
dtree.setMinSampleCount(5)

dtree.train(train_data, cv2.ml.ROW_SAMPLE, labels)
_, prediction = dtree.predict(test_data)
```

---

### 4. HOG Features (Histogram of Oriented Gradients)

**What it does**: Captures edge/gradient structure for object detection.

**HOG Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOG Feature Extraction                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Image → Gradient → Cells → Blocks → Feature Vector        │
│       ↓            ↓          ↓        ↓           ↓               │
│   [20×20]    [mag, angle] [5×5 hist] [2×2 cells] [324 features]   │
│                                                                     │
│   Each cell: histogram of gradient orientations (9 bins)           │
│   Blocks: overlapping groups of cells, normalized                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**HOG Parameters**:
| Parameter | Description |
|-----------|-------------|
| `winSize` | Detection window size |
| `blockSize` | Block size for normalization |
| `cellSize` | Cell size for histogram |
| `nbins` | Number of orientation bins (usually 9) |

**OpenCV**:
```python
# Custom HOG for digits
hog = cv2.HOGDescriptor(
    _winSize=(20, 20),
    _blockSize=(10, 10),
    _blockStride=(5, 5),
    _cellSize=(5, 5),
    _nbins=9
)
features = hog.compute(digit_image)

# Built-in pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(frame)
```

---

### 5. K-Means Clustering

**What it does**: Partitions N samples into K clusters, minimizing within-cluster variance.

**K-Means for Image Segmentation**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    K-Means Image Segmentation                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Original Image → Reshape to pixels → K-Means → Replace colors   │
│                                                                     │
│   [H×W×3] → [(H*W), 3] → cluster each pixel → K unique colors     │
│                                                                     │
│   Example (K=4):                                                    │
│   Each pixel assigned to one of 4 color clusters                   │
│   Result: simplified image with only 4 colors                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Elbow Method for K Selection**:
```
   Compactness
        │
   1000 │●
        │  ╲
    800 │   ●
        │    ╲
    600 │     ●
        │      ╲
    400 │       ● ← "Elbow" - choose K here
        │         ──●───●───●
        └────────────────────────
              2  3  4  5  6  7  K
```

**OpenCV**:
```python
# Reshape image for K-Means
pixels = image.reshape(-1, 3).astype(np.float32)

# Define criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Apply K-Means
compactness, labels, centers = cv2.kmeans(
    pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
)

# Reconstruct image
quantized = centers[labels.flatten()].reshape(image.shape)
```

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.ml.KNearest_create()` | Create KNN classifier |
| `cv2.ml.SVM_create()` | Create SVM classifier |
| `cv2.ml.DTrees_create()` | Create Decision Tree |
| `cv2.kmeans()` | K-Means clustering |
| `cv2.HOGDescriptor()` | Create HOG descriptor |
| `model.train()` | Train model |
| `model.predict()` | Make predictions |
| `model.save()` | Save model to file |
| `cv2.ml.SVM_load()` | Load SVM model |
| `cv2.ml.DTrees_load()` | Load Decision Tree |

---

## Digit Recognition Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Digit Recognition Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. Load digits.png (5000 samples, 20×20 each)                    │
│                           ↓                                         │
│   2. Flatten images: (5000, 20, 20) → (5000, 400)                  │
│                           ↓                                         │
│   3. Normalize: pixel values / 255.0                               │
│                           ↓                                         │
│   4. Split: 80% train, 20% test                                    │
│                           ↓                                         │
│   5. Train classifier (KNN/SVM/DTree)                              │
│                           ↓                                         │
│   6. Evaluate on test set                                          │
│                                                                     │
│   Typical Accuracy:                                                 │
│   - KNN (k=3):    ~96%                                             │
│   - SVM (RBF):    ~98%                                             │
│   - Decision Tree: ~85%                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Algorithm Comparison

| Algorithm | Training Speed | Prediction Speed | Accuracy | Interpretable |
|-----------|----------------|------------------|----------|---------------|
| KNN | Instant | Slow (large data) | Medium | Yes |
| SVM | Medium | Fast | High | No |
| Decision Tree | Fast | Fast | Medium | Yes |
| HOG+SVM | Medium | Fast | High | Partial |

---

## Practical Applications

| Application | Algorithm | Real-World Use |
|-------------|-----------|----------------|
| Digit OCR | KNN/SVM | Postal codes, checks |
| Pedestrian Detection | HOG+SVM | Autonomous vehicles, surveillance |
| Image Segmentation | K-Means | Photo editing, medical imaging |
| Document Classification | SVM | Email spam, sentiment analysis |

---

## Further Reading

- [OpenCV ML Tutorial](https://docs.opencv.org/4.x/d6/de2/tutorial_py_table_of_contents_ml.html)
- [SVM Tutorial](https://docs.opencv.org/4.x/d1/d73/tutorial_introduction_to_svm.html)
- [K-Means Tutorial](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html)
- [HOG Descriptor](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html)
