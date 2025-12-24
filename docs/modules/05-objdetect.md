---
layout: default
title: "05: Object Detection"
parent: Modules
nav_order: 5
permalink: /modules/05-objdetect
---

# Module 5: Object Detection
{: .fs-9 }

Classical object detection methods including Haar cascades and template matching.
{: .fs-6 .fw-300 }

---

## Topics Covered

- Haar cascade classifiers
- Face and eye detection
- Template matching
- Multi-scale detection

---

## Algorithm Explanations

### 1. Haar Cascade Pipeline

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

---

### 2. Haar-like Features

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
```

---

### 3. Integral Image

**Fast rectangle sum calculation**:
```
A───────B
│       │
│       │
D───────C

Sum = ii(C) - ii(B) - ii(D) + ii(A)
```

Only 4 array references regardless of rectangle size!

---

### 4. Cascade Rejection Process

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
│   └────┬─────┘                                                      │
│        │ 5000 pass                                                  │
│        ▼                                                            │
│   ┌──────────┐                                                      │
│   │ Stage 2  │  20 features, ~80% reject                           │
│   └────┬─────┘                                                      │
│        │ 1000 pass                                                  │
│        ▼                                                            │
│   ┌──────────┐                                                      │
│   │ Stage 3  │  50 features, ~90% reject                           │
│   └────┬─────┘                                                      │
│        │ 100 pass                                                   │
│        ▼                                                            │
│      .....                                                          │
│        │                                                            │
│        ▼                                                            │
│   5 DETECTIONS (faces found!)                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5. Template Matching

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
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Matching Methods**:

| Method | Range | Best Match |
|:-------|:------|:-----------|
| `TM_SQDIFF_NORMED` | [0, 1] | Minimum |
| `TM_CCORR_NORMED` | [0, 1] | Maximum |
| `TM_CCOEFF_NORMED` | [-1, 1] | Maximum |

---

### 6. Non-Maximum Suppression (NMS)

```
┌─────────────────────────────────────────────────────────────────────┐
│               Non-Maximum Suppression Process                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Before NMS                          After NMS                     │
│   (Multiple overlapping)              (Single best)                │
│                                                                     │
│   ┌───────────────┐                   ┌───────────────┐            │
│   │ ┌─────────┐   │                   │               │            │
│   │ │┌───────┐│   │                   │   ┌───────┐   │            │
│   │ ││ FACE ││   │       ──▶         │   │ FACE  │   │            │
│   │ │└───────┘│   │                   │   └───────┘   │            │
│   │ └─────────┘   │                   │               │            │
│   └───────────────┘                   └───────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**IoU (Intersection over Union)**:
```
IoU = Area(A ∩ B) / Area(A ∪ B)
```

---

## Pre-trained Cascades

| Cascade File | Detects |
|:-------------|:--------|
| `haarcascade_frontalface_default.xml` | Frontal faces |
| `haarcascade_profileface.xml` | Side profile faces |
| `haarcascade_eye.xml` | Eyes |
| `haarcascade_smile.xml` | Smiles |

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_cascade_classifiers.py` | Haar cascades, face/eye detection |
| `02_template_matching.py` | Template matching, multi-scale, NMS |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
| `cv2.CascadeClassifier(path)` | Load cascade |
| `cascade.detectMultiScale()` | Detect objects |
| `cv2.matchTemplate()` | Template matching |
| `cv2.minMaxLoc()` | Find best match |
| `cv2.dnn.NMSBoxes()` | Non-max suppression |

---

## Further Reading

- [Cascade Classifier Tutorial](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [Viola-Jones Paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
