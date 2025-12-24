---
layout: default
title: "Extras: Advanced Topics"
parent: Modules
nav_order: 12
permalink: /modules/extras
---

# Extra Modules: Advanced Topics
{: .fs-9 }

Advanced OpenCV features including face recognition, object tracking, and text detection.
{: .fs-6 .fw-300 }

{: .note }
Some features require `opencv-contrib-python` package.

---

## Topics Covered

- Face detection and recognition
- Object tracking algorithms
- Text detection and OCR

---

## 1. Face Recognition

### Face Recognition Pipeline

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

---

### Recognition Algorithms

#### Eigenfaces (PCA)

Projects faces into lower-dimensional eigenspace using Principal Component Analysis.

```python
recognizer = cv2.face.EigenFaceRecognizer_create(
    num_components=80,  # Number of eigenfaces
    threshold=10000     # Recognition threshold
)
```

#### Fisherfaces (LDA)

Maximizes between-class variance, minimizes within-class variance using Linear Discriminant Analysis.

```python
recognizer = cv2.face.FisherFaceRecognizer_create(
    num_components=0,   # 0 = use all
    threshold=10000
)
```

**Advantages over Eigenfaces**:
- Better handles lighting variations
- More discriminative features

---

#### LBPH (Local Binary Patterns Histograms)

Extracts texture features using local binary patterns.

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
└─────────────────────────────────────────────────────────────────────┘
```

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

## 2. Object Tracking

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

---

### Tracker Comparison

| Tracker | Speed | Accuracy | Occlusion | Description |
|:--------|:------|:---------|:----------|:------------|
| BOOSTING | Slow | Low | Poor | AdaBoost-based |
| MIL | Slow | Medium | Poor | Multiple Instance Learning |
| KCF | Fast | Medium | Poor | Kernelized Correlation Filters |
| TLD | Medium | Medium | Good | Tracking-Learning-Detection |
| MEDIANFLOW | Fast | High | Poor | Optical flow based |
| MOSSE | V.Fast | Low | Poor | Minimum Output Sum of Squared Error |
| CSRT | Medium | High | Medium | Discriminative Correlation Filter |
| GOTURN | Slow | High | Good | Deep learning (CNN) |

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

## 3. Text Detection and OCR

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

---

### MSER (Maximally Stable Extremal Regions)

Detects stable regions that often correspond to text.

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
```

---

### OCR with Tesseract

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
```

**PSM (Page Segmentation Mode)**:

| Value | Description |
|:------|:------------|
| 3 | Fully automatic page segmentation |
| 6 | Assume single uniform block of text |
| 7 | Treat image as single text line |
| 8 | Treat image as single word |
| 10 | Treat image as single character |

---

## Algorithm Comparison

### Face Recognition

| Method | Speed | Accuracy | Lighting | Update |
|:-------|:------|:---------|:---------|:-------|
| Eigenfaces | Fast | Low | Sensitive | No |
| Fisherfaces | Fast | Medium | Better | No |
| LBPH | Medium | Good | Robust | Yes |

### Trackers

| Tracker | Speed | Accuracy | Best For |
|:--------|:------|:---------|:---------|
| MOSSE | Fastest | Low | Simple tracking |
| KCF | Fast | Medium | Real-time |
| CSRT | Medium | High | Accurate tracking |
| GOTURN | Slow | High | Complex scenes |

### Text Detection

| Method | Speed | Accuracy | Scene Text |
|:-------|:------|:---------|:-----------|
| MSER | Fast | Medium | Limited |
| EAST | Medium | High | Good |
| Tesseract | Slow | High | Document |
| EasyOCR | Slow | High | General |

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_face_module.py` | Face detection, recognition (Eigenfaces, LBPH) |
| `02_tracking.py` | Single/multi-object tracking, tracker comparison |
| `03_text_ocr.py` | MSER, EAST, Tesseract integration |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
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
