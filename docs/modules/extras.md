---
layout: default
title: "Extras"
parent: Modules
nav_order: 12
---

# Extra Modules: Advanced Topics

Advanced OpenCV features including face recognition, object tracking, and text detection.

> **Note**: Some features require `opencv-contrib-python` package.

## Topics Covered

- Face detection and recognition
- Object tracking algorithms
- Text detection and OCR

---

## 1. Face Recognition

**Face Recognition Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Face Recognition System                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │  Input   │    │  Detect  │    │  Align   │    │ Extract  │    │
│   │  Image   │───▶│   Face   │───▶│   Face   │───▶│ Features │    │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘    │
│                                                         │          │
│                                                         ▼          │
│                                                   ┌──────────┐     │
│   ┌──────────┐                                    │  Match   │     │
│   │  Known   │───────────────────────────────────▶│ Against  │     │
│   │   Faces  │                                    │ Database │     │
│   │ Database │                                    └──────────┘     │
│   └──────────┘                                          │          │
│                                                         ▼          │
│                                                   ┌──────────┐     │
│                                                   │ Identity │     │
│                                                   │   or     │     │
│                                                   │ Unknown  │     │
│                                                   └──────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Face Detection with DNN

**Recommended approach** using deep learning:

```python
# Load SSD face detector
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Prepare input
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
net.setInput(blob)

# Detect faces
detections = net.forward()

# Process detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        x1, y1, x2, y2 = detections[0, 0, i, 3:7] * [w, h, w, h]
```

---

### Face Recognition Algorithms

#### Eigenfaces (PCA)

**What it does**: Projects faces into lower-dimensional eigenspace.

**Principal Component Analysis**:
```
1. Compute mean face: μ = (1/N) × Σᵢ xᵢ

2. Center the data: Φᵢ = xᵢ - μ

3. Compute covariance: C = (1/N) × Σᵢ ΦᵢΦᵢᵀ

4. Find eigenvectors of C: Cv = λv
   (Eigenfaces are eigenvectors with largest eigenvalues)

5. Project to eigenspace: ω = Uᵀ × (x - μ)
   Where U = matrix of top k eigenvectors
```

**Recognition**:
```
1. Project test face to eigenspace
2. Find nearest neighbor in projected training set
3. Use Euclidean distance: d = ||ω_test - ω_train||
```

```python
recognizer = cv2.face.EigenFaceRecognizer_create(
    num_components=80,  # Number of eigenfaces
    threshold=10000     # Recognition threshold
)
```

---

#### Fisherfaces (LDA)

**What it does**: Maximizes between-class variance, minimizes within-class.

**Linear Discriminant Analysis**:
```
Maximize: J(W) = (Wᵀ S_B W) / (Wᵀ S_W W)

Where:
  S_B = between-class scatter matrix
  S_W = within-class scatter matrix

S_B = Σᵢ Nᵢ × (μᵢ - μ)(μᵢ - μ)ᵀ
S_W = Σᵢ Σₓ∈Cᵢ (x - μᵢ)(x - μᵢ)ᵀ
```

**Advantages over Eigenfaces**:
- Better handles lighting variations
- More discriminative features

```python
recognizer = cv2.face.FisherFaceRecognizer_create(
    num_components=0,   # 0 = use all
    threshold=10000
)
```

---

#### LBPH (Local Binary Patterns Histograms)

**What it does**: Extracts texture features using local binary patterns.

**LBP Operator Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Local Binary Pattern (LBP)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Step 1: Get neighborhood        Step 2: Compare with center      │
│                                                                     │
│   ┌───┬───┬───┐                  ┌───┬───┬───┐                     │
│   │ 7 │ 9 │ 3 │                  │ 0 │ 1 │ 0 │   ≥5 → 1            │
│   ├───┼───┼───┤   center = 5     ├───┼───┼───┤   < 5 → 0            │
│   │ 6 │ 5 │ 2 │  ───────────▶   │ 1 │   │ 0 │                      │
│   ├───┼───┼───┤                  ├───┼───┼───┤                     │
│   │ 1 │ 8 │ 4 │                  │ 0 │ 1 │ 0 │                      │
│   └───┴───┴───┘                  └───┴───┴───┘                     │
│                                                                     │
│   Step 3: Read binary clockwise → 01001010 = 74 (decimal)          │
│                                                                     │
│           1                                                         │
│         ╱   ╲                                                       │
│        0     0      Binary: 01001010                               │
│        │     │      Decimal: 74                                    │
│        1     0      This becomes pixel value                       │
│         ╲   ╱                                                       │
│           0                                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**LBPH Face Histogram**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    LBPH Face Representation                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Face divided into grid:         Histogram per cell:              │
│                                                                     │
│   ┌───┬───┬───┬───┐               ┌────────────────┐               │
│   │ 1 │ 2 │ 3 │ 4 │               │  ▓░▓▓░▓░░▓    │ Cell 1        │
│   ├───┼───┼───┼───┤               ├────────────────┤               │
│   │ 5 │ 6 │ 7 │ 8 │               │  ░▓░▓▓░▓░     │ Cell 2        │
│   ├───┼───┼───┼───┤               ├────────────────┤               │
│   │ 9 │10 │11 │12 │   ─────▶      │  ▓▓░░▓░░▓     │ Cell 3        │
│   ├───┼───┼───┼───┤               ├────────────────┤               │
│   │13 │14 │15 │16 │               │     ...       │ ...           │
│   └───┴───┴───┴───┘               └────────────────┘               │
│                                          │                         │
│   8×8 grid = 64 cells                    │                         │
│                                          ▼                         │
│                    Concatenate all histograms → Feature Vector     │
│                    (64 cells × 256 bins = 16,384 features)         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**LBP Operator**:
```
For each pixel p with neighbors n₀...n₇:

LBP(p) = Σᵢ₌₀⁷ s(nᵢ - p) × 2ⁱ

Where s(x) = 1 if x ≥ 0, else 0

Result: 8-bit code (0-255) describing local texture
```

**Circular LBP**:
```
Sample P points on circle of radius R around center:
  xₚ = x + R × cos(2πp/P)
  yₚ = y + R × sin(2πp/P)
```

**Histogram Computation**:
```
1. Divide face into grid (e.g., 8×8 cells)
2. Compute LBP for each pixel
3. Build histogram for each cell
4. Concatenate histograms → feature vector
```

**Recognition**:
```
Compare histograms using Chi-squared distance:

χ²(H₁, H₂) = Σᵢ (H₁(i) - H₂(i))² / (H₁(i) + H₂(i))
```

```python
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,       # LBP radius
    neighbors=8,    # Number of neighbors
    grid_x=8,       # Grid cells in x
    grid_y=8,       # Grid cells in y
    threshold=80    # Recognition threshold
)
```

**Advantages**:
- Can be updated with new faces
- Robust to lighting changes
- Faster training

---

### Training and Prediction

```python
# Prepare training data
faces = [gray_face1, gray_face2, ...]  # Same size
labels = np.array([0, 0, 1, 1, 2, ...])  # Person IDs

# Train
recognizer.train(faces, labels)

# Predict
label, confidence = recognizer.predict(test_face)
# Lower confidence = better match

# Update (LBPH only)
recognizer.update(new_faces, new_labels)

# Save/Load
recognizer.save('model.yml')
recognizer.read('model.yml')
```

---

### Face Alignment

Normalize face orientation before recognition:

```python
def align_face(img, left_eye, right_eye):
    # Calculate rotation angle
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Eye center
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    # Rotate
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned
```

---

## 2. Object Tracking

**Object Tracking Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Object Tracking                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Frame 1            Frame 2            Frame 3            Frame N │
│                                                                     │
│   ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌────────┐│
│   │     ┌─┐   │     │       ┌─┐ │     │         ┌─┐     │    ┌─┐ ││
│   │     │●│   │────▶│       │●│ │────▶│         │●│────▶│    │●│ ││
│   │     └─┘   │     │       └─┘ │     │         └─┘     │    └─┘ ││
│   │           │     │           │     │           │     │        ││
│   └───────────┘     └───────────┘     └───────────┘     └────────┘│
│                                                                     │
│   Initialize        Predict new        Update model     Continuous │
│   with bbox         location           with appearance  tracking   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │ Tracking vs Detection:                                      │  │
│   │ • Detection: Find all objects each frame (slow but robust) │  │
│   │ • Tracking: Follow known object between frames (fast)      │  │
│   │ • Best: Combine both (detect periodically, track between)  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Tracker Types

| Tracker | Speed | Accuracy | Occlusion | Description |
|---------|-------|----------|-----------|-------------|
| BOOSTING | Slow | Low | Poor | AdaBoost-based |
| MIL | Slow | Medium | Poor | Multiple Instance Learning |
| KCF | Fast | Medium | Poor | Kernelized Correlation Filters |
| TLD | Medium | Medium | Good | Tracking-Learning-Detection |
| MEDIANFLOW | Fast | High | Poor | Optical flow based |
| MOSSE | V.Fast | Low | Poor | Minimum Output Sum of Squared Error |
| CSRT | Medium | High | Medium | Discriminative Correlation Filter |
| GOTURN | Slow | High | Good | Deep learning (CNN) |

---

### KCF (Kernelized Correlation Filters)

**What it does**: Tracks using correlation filters in Fourier domain.

**Correlation Filter**:
```
Train filter h that produces high response at target:

g = h ⊛ x

Where:
  x = image patch
  h = filter
  g = response map
  ⊛ = correlation
```

**Fourier Domain** (fast computation):
```
G = H* ⊙ X

Where:
  H* = complex conjugate of filter
  ⊙ = element-wise multiplication
```

**Kernel Trick**:
```
Use non-linear kernel for better separation:

k(x, x') = exp(-1/σ² × (||x||² + ||x'||² - 2F⁻¹(X* ⊙ X')))
```

---

### CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)

**Improvements over KCF**:
```
1. Spatial reliability map:
   - Learns which parts of target are most reliable
   - Reduces background interference

2. Channel reliability:
   - Weights different features (HOG, color)
   - Adapts to target appearance
```

---

### Tracking API

```python
# Create tracker
tracker = cv2.TrackerKCF_create()
# or
tracker = cv2.TrackerCSRT_create()
tracker = cv2.TrackerMIL_create()

# Initialize with bounding box
bbox = (x, y, width, height)
tracker.init(frame, bbox)

# Update in each frame
success, bbox = tracker.update(frame)

if success:
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

---

### Multi-Object Tracking

```python
# Manual approach (recommended)
trackers = []
for bbox in initial_bboxes:
    t = cv2.TrackerKCF_create()
    t.init(frame, bbox)
    trackers.append(t)

# Update all
for i, tracker in enumerate(trackers):
    success, bbox = tracker.update(frame)
    if success:
        # Draw bounding box
```

---

### Tracking + Detection Hybrid

**Best Practice**:
```
1. Detect objects periodically (every N frames)
2. Track between detections (fast)
3. Re-initialize trackers when detection available
4. Handle track-detection association (Hungarian algorithm)
```

---

## 3. Text Detection and OCR

**Text Detection and Recognition Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    OCR Pipeline                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Image           Text Detection         Text Recognition    │
│                                                                     │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐     │
│   │ Hello      │       │ ┌─────────┐ │       │             │     │
│   │   World    │  ──▶  │ │ Hello   │ │  ──▶  │  "Hello"    │     │
│   │            │       │ └─────────┘ │       │  "World"    │     │
│   │ OpenCV     │  ──▶  │ ┌─────────┐ │  ──▶  │  "OpenCV"   │     │
│   │            │       │ │ World   │ │       │             │     │
│   └─────────────┘       │ └─────────┘ │       └─────────────┘     │
│                         │ ┌─────────┐ │                           │
│                         │ │ OpenCV  │ │                           │
│                         │ └─────────┘ │                           │
│                         └─────────────┘                            │
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │ Methods:                                                  │    │
│   │ • MSER: Fast text region detection                       │    │
│   │ • EAST: Deep learning text detection (scene text)        │    │
│   │ • Tesseract: OCR engine for character recognition        │    │
│   │ • EasyOCR: All-in-one detection + recognition            │    │
│   └───────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### MSER (Maximally Stable Extremal Regions)

**What it does**: Detects stable regions that often correspond to text.

**Algorithm**:
```
1. Threshold image at all levels (0-255)
2. Track connected components through levels
3. Find regions that are "stable" (area changes slowly)

Stability criterion:
  q(i) = |Qᵢ₊Δ - Qᵢ₋Δ| / |Qᵢ|

Where Qᵢ = region at threshold i
```

```python
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray)

# Filter by aspect ratio and size
for region in regions:
    x, y, w, h = cv2.boundingRect(region)
    aspect = w / float(h)
    if 0.1 < aspect < 10 and w > 10 and h > 10:
        # Likely text region
```

---

### EAST Text Detector

**Efficient and Accurate Scene Text** detector using deep learning.

**Network Output**:
```
1. Score map: Probability of text at each location
2. Geometry: Rotated bounding box parameters
   - 4 distances (top, right, bottom, left)
   - 1 rotation angle
```

**Usage**:
```python
# Load model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Output layer names
outputLayers = ["feature_fusion/Conv_7/Sigmoid",  # Scores
                "feature_fusion/concat_3"]         # Geometry

# Prepare input (must be divisible by 32)
blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320),
                             (123.68, 116.78, 103.94),
                             swapRB=True, crop=False)

net.setInput(blob)
scores, geometry = net.forward(outputLayers)

# Decode and apply NMS
boxes, confidences = decode_predictions(scores, geometry)
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)
```

---

### OCR with Tesseract

**Integration**:
```python
import pytesseract
from PIL import Image

# Simple usage
text = pytesseract.image_to_string(image)

# With configuration
config = '--oem 3 --psm 6'
text = pytesseract.image_to_string(image, config=config)

# Get bounding boxes
boxes = pytesseract.image_to_boxes(image)

# Get detailed data
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
```

**OEM (OCR Engine Mode)**:
| Value | Description |
|-------|-------------|
| 0 | Legacy engine only |
| 1 | Neural nets LSTM only |
| 2 | Legacy + LSTM |
| 3 | Default (based on available) |

**PSM (Page Segmentation Mode)**:
| Value | Description |
|-------|-------------|
| 3 | Fully automatic page segmentation |
| 6 | Assume single uniform block of text |
| 7 | Treat image as single text line |
| 8 | Treat image as single word |
| 10 | Treat image as single character |

---

### OCR Preprocessing

```python
def preprocess_for_ocr(img):
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Noise removal
    denoised = cv2.fastNlMeansDenoising(gray)

    # 3. Thresholding
    _, binary = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Deskew (if needed)
    # 5. Rescale small text (2-3x)

    return binary
```

**Tips**:
- Remove noise before OCR
- Use adaptive threshold for uneven lighting
- Upscale small text
- Invert if dark background

---

### EasyOCR Alternative

```python
import easyocr

# Create reader
reader = easyocr.Reader(['en'])  # Languages

# Read text
results = reader.readtext(image)

# Results: [(bbox, text, confidence), ...]
for bbox, text, conf in results:
    print(f"{text} ({conf:.2f})")
```

**Advantages**:
- Easy to use
- 80+ languages
- Good accuracy out of box
- GPU support

---

## Algorithm Comparison

### Face Recognition

| Method | Speed | Accuracy | Lighting | Update |
|--------|-------|----------|----------|--------|
| Eigenfaces | Fast | Low | Sensitive | No |
| Fisherfaces | Fast | Medium | Better | No |
| LBPH | Medium | Good | Robust | Yes |

### Trackers

| Tracker | Speed | Accuracy | Best For |
|---------|-------|----------|----------|
| MOSSE | Fastest | Low | Simple tracking |
| KCF | Fast | Medium | Real-time |
| CSRT | Medium | High | Accurate tracking |
| GOTURN | Slow | High | Complex scenes |

### Text Detection

| Method | Speed | Accuracy | Scene Text |
|--------|-------|----------|------------|
| MSER | Fast | Medium | Limited |
| EAST | Medium | High | Good |
| Tesseract | Slow | High | Document |
| EasyOCR | Slow | High | General |

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_face_module.py` | Face detection, recognition (Eigenfaces, LBPH) |
| `02_tracking.py` | Single/multi-object tracking, tracker comparison |
| `03_text_ocr.py` | MSER, EAST, Tesseract integration |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.face.EigenFaceRecognizer_create()` | Eigenfaces recognizer |
| `cv2.face.FisherFaceRecognizer_create()` | Fisherfaces recognizer |
| `cv2.face.LBPHFaceRecognizer_create()` | LBPH recognizer |
| `recognizer.train()` | Train face recognizer |
| `recognizer.predict()` | Recognize face |
| `cv2.TrackerKCF_create()` | Create KCF tracker |
| `cv2.TrackerCSRT_create()` | Create CSRT tracker |
| `tracker.init()` | Initialize tracker |
| `tracker.update()` | Update tracker |
| `cv2.MSER_create()` | Create MSER detector |

---

## Further Reading

- [Face Recognition Tutorial](https://docs.opencv.org/4.x/dd/d65/classcv_1_1face_1_1FaceRecognizer.html)
- [Object Tracking](https://docs.opencv.org/4.x/d9/df8/group__tracking.html)
- [EAST Paper](https://arxiv.org/abs/1704.03155)
- [Tesseract Documentation](https://tesseract-ocr.github.io/)
