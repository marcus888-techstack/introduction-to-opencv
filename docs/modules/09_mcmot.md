---
layout: default
title: "09: Multi-Object Tracking"
parent: Modules
nav_order: 9
permalink: /modules/09-mcmot
---

# Module 9: Multi-Camera Multi-Object Tracking

Advanced tracking with person detection and re-identification across multiple cameras.

## Topics Covered

- OpenCV tracking API
- YOLOv4-tiny person detection
- Person re-identification (Re-ID)
- Multi-object tracking (MOT)
- Cross-camera tracking (MCMOT)

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_tracking_basics.py` | OpenCV tracking API, single object trackers |
| `02_person_detection.py` | YOLOv4-tiny person detection |
| `03_person_reid.py` | Person re-identification with deep features |
| `04_mot_tracker.py` | Multi-object tracking with SORT/DeepSORT concepts |
| `05_mcmot_multicam.py` | Cross-camera tracking and Re-ID matching |

---

## Key Concepts

### Tracking Pipeline
```
Detection → Feature Extraction → Association → Track Management
```

### Re-ID Matching
- Extract appearance features using deep networks
- Compute cosine similarity between feature vectors
- Match across cameras using appearance + spatial cues

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.TrackerCSRT_create()` | Create CSRT tracker |
| `cv2.TrackerKCF_create()` | Create KCF tracker |
| `cv2.dnn.readNet()` | Load YOLO/Re-ID models |
| `cv2.dnn.blobFromImage()` | Prepare input for DNN |
| `tracker.init()` | Initialize tracker |
| `tracker.update()` | Update tracker position |

---

## Further Reading

- [OpenCV Object Tracking](https://docs.opencv.org/4.x/d9/df8/group__tracking.html)
- [SORT Algorithm Paper](https://arxiv.org/abs/1602.00763)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
