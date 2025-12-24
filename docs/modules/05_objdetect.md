---
layout: default
title: "05: Object Detection"
parent: Modules
nav_order: 5
permalink: /modules/05-objdetect
---

# Module 5: Object Detection

Classical object detection methods including Haar cascades and template matching.

## Topics Covered

- Haar cascade classifiers
- Face and eye detection
- Template matching
- Multi-scale detection

---

## Algorithm Explanations

### 1. Haar Cascade Classifiers

**What it does**: Detects objects using a cascade of weak classifiers trained on Haar-like features.

**Haar Cascade Pipeline Overview**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                   Haar Cascade Detection Pipeline                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Image                                                       │
│   ┌───────────────────────────────────┐                             │
│   │                                   │                             │
│   │   ┌───┐  ┌───┐  ┌───┐  ┌───┐     │  Sliding window at         │
│   │   │   │  │   │  │   │  │   │ ... │  multiple scales            │
│   │   └───┘  └───┘  └───┘  └───┘     │                             │
│   │                                   │                             │
│   └───────────────────────────────────┘                             │
│              │                                                      │
│              ▼                                                      │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │  Cascade of Classifiers                                 │      │
│   │  Stage 1 → Stage 2 → Stage 3 → ... → Stage N → DETECT  │      │
│   │     ↓          ↓          ↓                             │      │
│   │  Reject     Reject     Reject      (Most windows        │      │
│   │                                     rejected early!)     │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Haar-like Features

Haar features capture intensity differences between adjacent regions:

```
Edge Features:
┌───┬───┐  ┌───────┐
│ + │ - │  │   +   │
└───┴───┘  ├───────┤
           │   -   │
           └───────┘

Line Features:
┌───┬───┬───┐  ┌───────┐
│ - │ + │ - │  │   -   │
└───┴───┴───┘  ├───────┤
               │   +   │
               ├───────┤
               │   -   │
               └───────┘

Four-Rectangle Feature:
┌───┬───┐
│ + │ - │
├───┼───┤
│ - │ + │
└───┴───┘
```

**Feature Value Calculation**:
```
f = Σ(pixels in white) - Σ(pixels in black)
```

#### Integral Image

**What it does**: Enables O(1) calculation of any rectangular sum.

**Formula**:
```
ii(x, y) = Σₓ'≤ₓ Σᵧ'≤ᵧ i(x', y')
```

**Integral Image Visualization**:
```
Original Image                    Integral Image

┌───┬───┬───┬───┐                ┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │                │ 1 │ 3 │ 6 │10 │
├───┼───┼───┼───┤                ├───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │      ──▶       │ 6 │14 │24 │36 │
├───┼───┼───┼───┤                ├───┼───┼───┼───┤
│ 9 │10 │11 │12 │                │15 │33 │54 │78 │
└───┴───┴───┴───┘                └───┴───┴───┴───┘

ii(x,y) = sum of ALL pixels above and to the left
```

**Sum of rectangle ABCD**:
```
A───────B
│       │
│       │
D───────C

Sum = ii(C) - ii(B) - ii(D) + ii(A)
```

**Visual Proof**:
```
┌───────────┬───────────┐
│     A     │     B     │
│   (area   │  (area to │
│  to left  │   remove) │
│  & above) │           │
├───────────┼───────────┤
│     D     │ RECTANGLE │
│  (area    │ ████████ │
│  to       │ ████████ │
│  remove)  │ (wanted!) │
└───────────┴───────────┘

ii(C) includes everything
- ii(B) removes top area
- ii(D) removes left area
+ ii(A) adds back corner (removed twice)
= Rectangle sum!
```

Only 4 array references regardless of rectangle size!

#### AdaBoost Training

**What it does**: Selects best weak classifiers and combines them.

**Algorithm**:
```
1. Initialize weights: wᵢ = 1/N

2. For t = 1 to T:
   a. Train all weak classifiers on weighted samples
   b. Select classifier hₜ with lowest weighted error εₜ
   c. Compute weight: αₜ = ½ ln((1-εₜ)/εₜ)
   d. Update weights:
      wᵢ ← wᵢ × exp(-αₜ × yᵢ × hₜ(xᵢ))
   e. Normalize weights

3. Final classifier:
   H(x) = sign(Σₜ αₜ × hₜ(x))
```

#### Cascade Structure

**What it does**: Chain of stages that quickly rejects non-objects.

**Cascade Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cascade Rejection Process                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   10000 windows (candidate regions)                                 │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────┐                                                      │
│   │ Stage 1  │  5 features, ~50% reject                            │
│   │ (simple) │                                                      │
│   └────┬─────┘                                                      │
│        │ 5000 pass                                                  │
│        ▼                                                            │
│   ┌──────────┐                                                      │
│   │ Stage 2  │  20 features, ~80% reject                           │
│   │          │                                                      │
│   └────┬─────┘                                                      │
│        │ 1000 pass                                                  │
│        ▼                                                            │
│   ┌──────────┐                                                      │
│   │ Stage 3  │  50 features, ~90% reject                           │
│   │          │                                                      │
│   └────┬─────┘                                                      │
│        │ 100 pass                                                   │
│        ▼                                                            │
│      .....                                                          │
│        │ 10 pass                                                    │
│        ▼                                                            │
│   ┌──────────┐                                                      │
│   │ Stage N  │  200+ features (thorough check)                     │
│   │ (complex)│                                                      │
│   └────┬─────┘                                                      │
│        │                                                            │
│        ▼                                                            │
│   5 DETECTIONS (faces found!)                                       │
│                                                                     │
│   Key insight: Most non-faces rejected by simple Stage 1           │
│                Complex stages only run on likely candidates         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Design**:
- Early stages: few features, high false positive rate
- Later stages: more features, lower false positive rate
- Overall: high detection rate, low false positive rate

**Cascade Properties**:
```
Detection Rate = Π(dᵢ)  (product of stage detection rates)
False Positive Rate = Π(fᵢ)  (product of stage FP rates)
```

#### Multi-scale Detection

**Image Pyramid for Scale Invariance**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Scale Detection                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Original (scale 1.0)      Scale 0.9          Scale 0.81          │
│   ┌───────────────────┐     ┌────────────────┐  ┌──────────────┐   │
│   │                   │     │                │  │              │   │
│   │    ┌────┐         │     │   ┌────┐       │  │  ┌────┐      │   │
│   │    │face│         │     │   │face│       │  │  │face│      │   │
│   │    └────┘         │     │   └────┘       │  │  └────┘      │   │
│   │                   │     │                │  │              │   │
│   │  Fixed 24×24      │     │                │  │              │   │
│   │  detector window  │     │                │  │              │   │
│   └───────────────────┘     └────────────────┘  └──────────────┘   │
│                                                                     │
│   Large face detected       Medium face        Small face          │
│   at scale 1.0              at scale 0.9       at scale 0.81       │
│                                                                     │
│   scaleFactor = 1.1 means: new_size = old_size / 1.1               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Algorithm**:
```
1. Create image pyramid by scaling down
2. Apply detector at each scale
3. Map detections back to original size
4. Apply Non-Maximum Suppression (NMS)
```

**Parameters**:
- `scaleFactor`: How much to reduce image each iteration
- `minNeighbors`: Minimum overlapping detections required
- `minSize`, `maxSize`: Detection size limits

---

### 2. detectMultiScale

**OpenCV Function**:
```python
objects = cascade.detectMultiScale(
    image,
    scaleFactor=1.1,    # Image size reduction per scale
    minNeighbors=5,     # Required neighbor detections
    flags=0,
    minSize=(30, 30),
    maxSize=(300, 300)
)
```

**Parameter Tuning**:
| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| scaleFactor | More accurate, slower | Faster, may miss |
| minNeighbors | More detections, more false positives | Fewer detections, more reliable |

---

### 3. Template Matching

**What it does**: Finds location of a template image within a larger image.

**Template Matching Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Template Matching                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Template         Search Image              Result Map            │
│   ┌─────┐         ┌─────────────────┐        ┌─────────────────┐   │
│   │ ABC │         │                 │        │  . . . . . . .  │   │
│   └─────┘         │     ABC  ○      │   ──▶  │  . . ●   . . .  │   │
│                   │                 │        │  . . . . . . .  │   │
│   Slide template  │                 │        │                 │   │
│   across image    └─────────────────┘        └─────────────────┘   │
│                                              ● = Best match        │
│   At each position, compute similarity                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Sliding Window Operation**:
```
Step 1        Step 2        Step 3        ...

┌─────────┐   ┌─────────┐   ┌─────────┐
│┌───┐    │   │ ┌───┐   │   │  ┌───┐  │
││ T │    │   │ │ T │   │   │  │ T │  │
│└───┘    │   │ └───┘   │   │  └───┘  │
│         │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘

Compute        Compute       Compute
R(0,0)         R(1,0)        R(2,0)

Result: R(x,y) = similarity at position (x,y)
```

#### Matching Methods

**Squared Difference** (`TM_SQDIFF`):
```
R(x,y) = Σₓ',ᵧ' [T(x',y') - I(x+x', y+y')]²
```
Best match: **minimum** value

**Normalized Squared Difference** (`TM_SQDIFF_NORMED`):
```
R(x,y) = Σ[T(x',y') - I(x+x', y+y')]² / √(Σ T(x',y')² × Σ I(x+x', y+y')²)
```
Range: [0, 1], best match: **minimum**

**Cross-Correlation** (`TM_CCORR`):
```
R(x,y) = Σₓ',ᵧ' T(x',y') × I(x+x', y+y')
```
Best match: **maximum** value

**Normalized Cross-Correlation** (`TM_CCORR_NORMED`):
```
R(x,y) = Σ[T(x',y') × I(x+x', y+y')] / √(Σ T(x',y')² × Σ I(x+x', y+y')²)
```
Range: [0, 1], best match: **maximum**

**Correlation Coefficient** (`TM_CCOEFF`):
```
R(x,y) = Σₓ',ᵧ' T'(x',y') × I'(x+x', y+y')

Where:
T'(x',y') = T(x',y') - mean(T)
I'(x,y) = I(x,y) - mean(I_patch)
```
Best match: **maximum** value

**Normalized Correlation Coefficient** (`TM_CCOEFF_NORMED`):
```
R(x,y) = Σ T' × I' / √(Σ T'² × Σ I'²)
```
Range: [-1, 1], best match: **maximum** (1 = perfect match)

**OpenCV**:
```python
result = cv2.matchTemplate(image, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
```

---

### 4. Multi-Scale Template Matching

**Problem**: Template matching is not scale-invariant.

**Solution**: Search across multiple scales:

```python
for scale in np.linspace(0.5, 2.0, 20):
    resized = cv2.resize(template, None, fx=scale, fy=scale)
    result = cv2.matchTemplate(image, resized, method)
    # Track best match across scales
```

---

### 5. Non-Maximum Suppression (NMS)

**What it does**: Removes overlapping detections, keeping only the best.

**NMS Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│               Non-Maximum Suppression Process                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Before NMS                          After NMS                     │
│   (Multiple overlapping detections)   (Single best detection)      │
│                                                                     │
│   ┌───────────────┐                   ┌───────────────┐            │
│   │ ┌─────────┐   │                   │               │            │
│   │ │┌───────┐│   │                   │   ┌───────┐   │            │
│   │ ││ FACE ││   │       ──▶         │   │ FACE  │   │            │
│   │ │└───────┘│   │                   │   └───────┘   │            │
│   │ └─────────┘   │                   │               │            │
│   │  └─────────┘  │                   │               │            │
│   └───────────────┘                   └───────────────┘            │
│                                                                     │
│   conf: 0.95, 0.92, 0.88             Only 0.95 remains             │
│   (overlapping boxes)                (suppressed others)            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Algorithm**:
```
1. Sort detections by confidence
2. Pick the highest confidence detection
3. Remove all detections with IoU > threshold
4. Repeat until no detections remain
```

**Intersection over Union (IoU)**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    IoU Calculation                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    Box A            Box B           Intersection        Union      │
│   ┌───────┐                                                        │
│   │       │       ┌───────┐                                        │
│   │   ┌───┼───────┤       │       ┌───┐          ┌───────────────┐│
│   │   │///│///////│       │  =    │///│    /     │               ││
│   └───┼───┘       │       │       └───┘          │               ││
│       │           │       │                      │               ││
│       └───────────┘       │                      └───────────────┘│
│                                                                     │
│   IoU = Area(A ∩ B) / Area(A ∪ B)                                  │
│                                                                     │
│   IoU = 1.0  → Perfect overlap (same box)                          │
│   IoU = 0.0  → No overlap                                          │
│   IoU > 0.5  → Significant overlap (typically suppressed)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
IoU = Area(A ∩ B) / Area(A ∪ B)
    = Area(A ∩ B) / (Area(A) + Area(B) - Area(A ∩ B))
```

**OpenCV** (for rectangles):
```python
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
```

---

## Comparison

| Method | Speed | Scale Invariant | Rotation Invariant | Accuracy |
|--------|-------|-----------------|-------------------|----------|
| Haar Cascade | Fast | Yes (multi-scale) | Limited | Medium |
| Template Matching | Slow | No | No | High (exact match) |
| Multi-scale Template | Slower | Yes | No | High |

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_cascade_classifiers.py` | Haar cascades, face/eye detection |
| `02_template_matching.py` | Template matching, multi-scale, NMS |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.CascadeClassifier(path)` | Load cascade |
| `cascade.detectMultiScale()` | Detect objects |
| `cv2.matchTemplate()` | Template matching |
| `cv2.minMaxLoc()` | Find best match |
| `cv2.dnn.NMSBoxes()` | Non-max suppression |

---

## Pre-trained Cascades

| Cascade File | Detects |
|--------------|---------|
| `haarcascade_frontalface_default.xml` | Frontal faces |
| `haarcascade_frontalface_alt.xml` | Frontal faces (alternative) |
| `haarcascade_profileface.xml` | Side profile faces |
| `haarcascade_eye.xml` | Eyes |
| `haarcascade_smile.xml` | Smiles |
| `haarcascade_fullbody.xml` | Full body |
| `haarcascade_frontalcatface.xml` | Cat faces |

---

## Further Reading

- [Cascade Classifier Tutorial](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [Viola-Jones Paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
- [Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
