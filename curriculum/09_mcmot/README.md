# Module 10: Multi-Camera Multi-Object Tracking (MCMOT)

## Overview

This module covers Multi-Object Tracking (MOT) and Multi-Camera Multi-Object Tracking (MCMOT) using OpenCV. The **core focus is Person Re-Identification (Re-ID)**, which enables tracking the same person across different frames, through occlusions, and across multiple camera views.

## Topics Covered

1. **Tracking Fundamentals** - OpenCV trackers, IoU, tracking-by-detection
2. **Person Detection** - YOLO with OpenCV DNN for person detection
3. **Person Re-Identification (Re-ID)** - Feature extraction and appearance matching (CORE)
4. **MOT with Re-ID** - SORT/DeepSORT-style tracking with appearance features
5. **Multi-Camera Tracking** - Cross-camera Re-ID and global track management

## Prerequisites

- Module 06: Video Processing (VideoCapture, frame handling)
- Module 08: DNN (cv2.dnn module basics)
- Understanding of NumPy arrays

Required packages:
```bash
pip install scipy  # For Hungarian algorithm
```

---

## 1. Tracking vs Detection

### When to Detect vs Track

```
Detection:                          Tracking:
┌──────────────────────────┐       ┌──────────────────────────┐
│ Every Frame              │       │ Frame-to-Frame           │
│                          │       │                          │
│  - Independent frames    │       │  - Temporal continuity   │
│  - More computation      │       │  - Less computation      │
│  - No ID persistence     │       │  - ID persistence        │
│  - Handles new objects   │       │  - Can lose track        │
│                          │       │                          │
└──────────────────────────┘       └──────────────────────────┘

Best Practice: Tracking-by-Detection
┌─────────────────────────────────────────────────────────────┐
│  Detect periodically → Track between detections → Re-ID    │
│                                                             │
│  Frame 1: Detect + Assign IDs                               │
│  Frame 2: Track (fast)                                      │
│  Frame 3: Track (fast)                                      │
│  Frame 4: Detect + Re-associate IDs                         │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

### OpenCV Built-in Trackers

| Tracker | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| `cv2.TrackerMOSSE_create()` | Very Fast | Low | Real-time, low accuracy OK |
| `cv2.TrackerKCF_create()` | Fast | Medium | General purpose |
| `cv2.TrackerCSRT_create()` | Slow | High | Accuracy critical |

---

## 2. Person Re-Identification (Re-ID) - CORE CONCEPT

Person Re-ID is the task of matching the same person across different images, camera views, or time instances based on **appearance features**.

### Re-ID Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Person Re-ID Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   INPUT                    MODEL                    OUTPUT               │
│  ┌────────┐             ┌─────────┐             ┌────────────────┐      │
│  │ Person │             │  Re-ID  │             │ Feature Vector │      │
│  │  Crop  │  ────────>  │   CNN   │  ────────>  │   (512-dim)    │      │
│  │128x256 │             │ (ONNX)  │             │ [0.2, -0.1,...]│      │
│  └────────┘             └─────────┘             └────────────────┘      │
│                                                                          │
│  Person A ──> [0.82, 0.15, -0.33, ...]  ─┐                              │
│                                           │── cosine_distance = 0.12    │
│  Person A ──> [0.79, 0.18, -0.30, ...]  ─┘   (SAME person!)             │
│  (different                                                              │
│   angle)                                                                 │
│                                                                          │
│  Person B ──> [-0.45, 0.62, 0.21, ...]  ─┐                              │
│                                           │── cosine_distance = 0.89    │
│  Person A ──> [0.82, 0.15, -0.33, ...]  ─┘   (DIFFERENT persons)        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Feature Embedding Space

Similar persons cluster together in the high-dimensional embedding space:

```
                    Feature Space Visualization (2D projection)

                         ▲ Dimension 2
                         │
                         │    ○ ○           Person A appearances
                         │   ○   ○          (clustered together)
                         │    ○ ○
                         │
                         │                    □ □
                         │                   □   □   Person B
                         │                    □ □
                         │
                         │        △ △
                         │       △   △        Person C
                         │        △ △
                         │
         ────────────────┼────────────────────────► Dimension 1
                         │

    Same person = Close in embedding space (low cosine distance)
    Different persons = Far apart (high cosine distance)
```

### Cosine Similarity & Distance

```
Cosine Similarity:
                        A · B           sum(A[i] * B[i])
    similarity(A, B) = ─────── = ───────────────────────────
                       |A||B|    sqrt(sum(A²)) * sqrt(sum(B²))

    Range: [-1, 1]

Cosine Distance:
    distance(A, B) = 1 - similarity(A, B)

    Range: [0, 2]

    distance < 0.5  →  Likely SAME person
    distance > 0.7  →  Likely DIFFERENT persons

Example:
    feat_A = [0.82, 0.15, -0.33, 0.45, ...]
    feat_B = [0.79, 0.18, -0.30, 0.42, ...]

    similarity = 0.98
    distance = 0.02  →  Very likely same person!
```

### Gallery vs Query Matching

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Gallery-Query Re-ID Matching                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GALLERY (Known Persons)              QUERY (Unknown Person)            │
│  ┌──────────────────────┐             ┌──────────────────────┐          │
│  │ ID=1: [feat_1]       │             │ New detection:       │          │
│  │ ID=2: [feat_2]       │    Match    │ [query_feat]         │          │
│  │ ID=3: [feat_3]       │ ◄────────── │                      │          │
│  │ ID=4: [feat_4]       │             │ Who is this?         │          │
│  │ ...                  │             │                      │          │
│  └──────────────────────┘             └──────────────────────┘          │
│                                                                          │
│  Matching Process:                                                       │
│  1. Compute distance to each gallery entry                              │
│  2. Find minimum distance                                               │
│  3. If min_distance < threshold → Match to that ID                      │
│  4. If min_distance > threshold → New person (assign new ID)            │
│                                                                          │
│  Example:                                                               │
│    distance(query, ID=1) = 0.82                                         │
│    distance(query, ID=2) = 0.15  ← Minimum!                             │
│    distance(query, ID=3) = 0.91                                         │
│    distance(query, ID=4) = 0.67                                         │
│                                                                          │
│    min_distance = 0.15 < threshold(0.5) → Query matches ID=2            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. MOT with Re-ID (DeepSORT-style)

Multi-Object Tracking combines **motion prediction** with **appearance matching**.

### SORT Algorithm (Simple Online Realtime Tracking)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SORT Algorithm                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each frame:                                                        │
│                                                                          │
│  1. PREDICT: Use Kalman Filter to predict next positions                │
│     ┌─────────────────────────────────────────────────────────┐         │
│     │ Track 1: [100, 50]  →  predict  →  [105, 52]            │         │
│     │ Track 2: [200, 80]  →  predict  →  [198, 82]            │         │
│     └─────────────────────────────────────────────────────────┘         │
│                                                                          │
│  2. DETECT: Get new detections from detector                            │
│     ┌─────────────────────────────────────────────────────────┐         │
│     │ Detection A: [106, 51]                                   │         │
│     │ Detection B: [300, 100]  (new person)                    │         │
│     │ Detection C: [197, 83]                                   │         │
│     └─────────────────────────────────────────────────────────┘         │
│                                                                          │
│  3. ASSOCIATE: Match predictions to detections (Hungarian)              │
│     ┌─────────────────────────────────────────────────────────┐         │
│     │ Cost Matrix (IoU-based):                                 │         │
│     │           Det A   Det B   Det C                          │         │
│     │ Track 1 [ 0.85   0.02    0.05 ]                         │         │
│     │ Track 2 [ 0.03   0.01    0.82 ]                         │         │
│     │                                                          │         │
│     │ Optimal assignment: Track1→DetA, Track2→DetC            │         │
│     │ Unmatched: Det B → New Track 3                           │         │
│     └─────────────────────────────────────────────────────────┘         │
│                                                                          │
│  4. UPDATE: Correct Kalman Filter with matched detections               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### DeepSORT: Adding Appearance (Re-ID)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DeepSORT: Motion + Appearance                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Cost Matrix = λ₁ × IoU_cost + λ₂ × Appearance_cost                     │
│                                                                          │
│  ┌────────────────────────────┐   ┌────────────────────────────┐        │
│  │     IoU Cost Matrix        │   │   Appearance Cost Matrix    │        │
│  │     (Spatial overlap)      │ + │   (Re-ID cosine distance)  │        │
│  │                            │   │                            │        │
│  │       Det A  Det B  Det C  │   │       Det A  Det B  Det C  │        │
│  │ Trk1 [ 0.2   0.9    0.8 ]  │   │ Trk1 [ 0.1   0.8    0.7 ]  │        │
│  │ Trk2 [ 0.85  0.95   0.15]  │   │ Trk2 [ 0.6   0.9    0.2 ]  │        │
│  │                            │   │                            │        │
│  └────────────────────────────┘   └────────────────────────────┘        │
│                                                                          │
│  Combined (λ₁=0.5, λ₂=0.5):                                             │
│  ┌────────────────────────────┐                                         │
│  │    Combined Cost Matrix    │                                         │
│  │       Det A  Det B  Det C  │                                         │
│  │ Trk1 [ 0.15  0.85   0.75]  │                                         │
│  │ Trk2 [ 0.73  0.93   0.18]  │   ← Best matches via Hungarian          │
│  └────────────────────────────┘                                         │
│                                                                          │
│  Why Appearance Helps:                                                  │
│  - Handles occlusion (person hidden, then reappears)                    │
│  - Handles ID switches when paths cross                                 │
│  - Re-associates after missed detections                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Track Lifecycle

```
Track States:
┌───────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  TENTATIVE ────────────> CONFIRMED ────────────> DELETED              │
│     │                        │                       ▲                │
│     │   (n_init hits)        │   (max_age missed)    │                │
│     │                        │                       │                │
│     └────────────────────────┼───────────────────────┘                │
│            (missed early)    │                                        │
│                              │                                        │
│                              ▼                                        │
│                           UPDATE                                      │
│                      (each matched                                    │
│                       detection)                                      │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘

Parameters:
- n_init = 3      # Hits needed to confirm track
- max_age = 30    # Frames before deleting unmatched track
```

---

## 4. Multi-Camera Multi-Object Tracking (MCMOT)

MCMOT extends MOT to handle multiple camera views simultaneously.

### Camera Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multi-Camera Setup Example                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│     Camera 1                    Camera 2                 Camera 3       │
│   ┌──────────┐               ┌──────────┐             ┌──────────┐     │
│   │  Entry   │               │ Hallway  │             │  Exit    │     │
│   │  Hall    │ ───────────>  │          │ ─────────>  │          │     │
│   │          │   Person      │          │   Person    │          │     │
│   │  ID: 1   │   walks       │  ID: ?   │   exits     │  ID: ?   │     │
│   └──────────┘               └──────────┘             └──────────┘     │
│                                                                          │
│   Challenge: Same person must have same ID across all cameras           │
│                                                                          │
│   Solution: Cross-Camera Re-ID                                          │
│   - Each camera runs local MOT                                          │
│   - Re-ID matches local tracks to global gallery                        │
│   - Global ID assigned based on appearance match                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### MCMOT Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MCMOT System Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Camera 1          Camera 2          Camera 3                           │
│     │                 │                 │                               │
│     ▼                 ▼                 ▼                               │
│  ┌──────┐          ┌──────┐          ┌──────┐                          │
│  │ MOT  │          │ MOT  │          │ MOT  │      Local Tracking       │
│  │Local │          │Local │          │Local │      (per camera)         │
│  └──┬───┘          └──┬───┘          └──┬───┘                          │
│     │                 │                 │                               │
│     │ local_id=1      │ local_id=3      │ local_id=7                   │
│     │ features        │ features        │ features                      │
│     │                 │                 │                               │
│     ▼                 ▼                 ▼                               │
│  ┌──────────────────────────────────────────────────────┐              │
│  │                  Global Re-ID Manager                 │              │
│  │                                                       │              │
│  │  Global Gallery:                                      │              │
│  │  ┌─────────────────────────────────────────────────┐ │              │
│  │  │ Global ID=1: features (updated from all cams)   │ │              │
│  │  │ Global ID=2: features                           │ │              │
│  │  │ Global ID=3: features                           │ │              │
│  │  └─────────────────────────────────────────────────┘ │              │
│  │                                                       │              │
│  │  Local-to-Global Mapping:                            │              │
│  │  Cam1:local_1 → Global_1                             │              │
│  │  Cam2:local_3 → Global_1  (same person!)             │              │
│  │  Cam3:local_7 → Global_1  (same person!)             │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                          │
│                         │                                               │
│                         ▼                                               │
│                  ┌─────────────┐                                        │
│                  │   Output    │                                        │
│                  │ Global IDs  │                                        │
│                  │ All Cameras │                                        │
│                  └─────────────┘                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cross-Camera Re-ID Matching

```
Cross-Camera Matching Process:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  1. New track appears in Camera 2 (local_id = 5)                        │
│                                                                          │
│  2. Extract Re-ID features: feat_new = [0.82, 0.15, ...]                │
│                                                                          │
│  3. Compare against Global Gallery:                                      │
│     ┌─────────────────────────────────────────────────────┐             │
│     │ Global_1: dist = 0.18  ← Below threshold!           │             │
│     │ Global_2: dist = 0.85                               │             │
│     │ Global_3: dist = 0.72                               │             │
│     └─────────────────────────────────────────────────────┘             │
│                                                                          │
│  4. Decision:                                                           │
│     - min_distance = 0.18 < threshold(0.5)                              │
│     - Match! Cam2:local_5 → Global_1                                    │
│                                                                          │
│  5. Update Global_1 features with exponential moving average:           │
│     feat_global = 0.9 * feat_global + 0.1 * feat_new                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tutorial Files

| File | Lines | Description |
|------|-------|-------------|
| `01_tracking_basics.py` | ~300 | OpenCV trackers, IoU, tracking concepts |
| `02_person_detection.py` | ~350 | YOLO person detection with cv2.dnn |
| **`03_person_reid.py`** | ~500 | **Person Re-ID - CORE MODULE** |
| `04_mot_tracker.py` | ~450 | MOT with Kalman + Re-ID (DeepSORT-style) |
| `05_mcmot_multicam.py` | ~500 | Multi-camera tracking with cross-camera Re-ID |

### Learning Path

```
01_tracking_basics   →  Understand single-object tracking & why MOT is needed
        │
        ▼
02_person_detection  →  Detect persons with YOLO, output bounding boxes
        │
        ▼
03_person_reid ★     →  CORE: Extract features, match appearances
        │
        ▼
04_mot_tracker       →  Combine motion (Kalman) + appearance (Re-ID)
        │
        ▼
05_mcmot_multicam    →  Apply Re-ID across multiple cameras
```

---

## Key OpenCV Functions

### Detection & Tracking

| Function | Purpose |
|----------|---------|
| `cv2.TrackerCSRT_create()` | Create CSRT tracker (accurate) |
| `cv2.TrackerKCF_create()` | Create KCF tracker (fast) |
| `cv2.dnn.readNet(weights, cfg)` | Load YOLO network |
| `cv2.dnn.readNetFromONNX(path)` | Load ONNX model (Re-ID) |
| `cv2.dnn.blobFromImage()` | Preprocess image for DNN |
| `cv2.dnn.NMSBoxes()` | Non-maximum suppression |

### Motion & Filtering

| Function | Purpose |
|----------|---------|
| `cv2.KalmanFilter(dim_state, dim_meas)` | Create Kalman filter |
| `kalman.predict()` | Predict next state |
| `kalman.correct(measurement)` | Update with measurement |

### Image Processing

| Function | Purpose |
|----------|---------|
| `cv2.resize()` | Resize person crop for Re-ID |
| `cv2.rectangle()` | Draw bounding box |
| `cv2.putText()` | Draw track ID |

### NumPy Operations (for Re-ID)

| Operation | Purpose |
|-----------|---------|
| `np.dot(a, b)` | Dot product for cosine similarity |
| `np.linalg.norm(a)` | L2 norm for normalization |
| `np.argmin(distances)` | Find best match |

### External (scipy)

| Function | Purpose |
|----------|---------|
| `scipy.optimize.linear_sum_assignment()` | Hungarian algorithm for optimal matching |

---

## Models Used

### YOLO v4-tiny (Person Detection)

```
Files needed:
- yolov4-tiny.weights  (23 MB)
- yolov4-tiny.cfg
- coco.names

Usage:
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
```

### Person Re-ID Model (OpenCV Zoo)

```
File: person_reid_youtu_2021nov.onnx

Input:  128 x 256 RGB image (person crop)
Output: 512-dimensional feature vector

Usage:
reid_net = cv2.dnn.readNetFromONNX("person_reid_youtu_2021nov.onnx")
```

Download models:
```bash
cd curriculum/sample_data
python download_samples.py
```

---

## Performance Tips

1. **Detection Frequency**: Detect every 3-5 frames, track in between
2. **Re-ID Batch Processing**: Extract features in batches for efficiency
3. **GPU Acceleration**: Use `net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)`
4. **Feature Caching**: Store last N features per track, average for matching
5. **Early Rejection**: Skip Re-ID if IoU is very high (clearly same object)

---

## References

- [OpenCV MOT Blog](https://opencv.org/blog/multiple-object-tracking-in-realtime/)
- [SORT Paper](https://arxiv.org/abs/1602.00763)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [OpenCV Zoo - Person Re-ID](https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtu)
- [OpenCV DNN Module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
