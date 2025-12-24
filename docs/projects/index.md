---
layout: default
title: Projects
nav_order: 4
has_children: true
permalink: /projects
---

# Practical Projects
{: .fs-9 }

Build 6 real-world computer vision applications that solve practical problems.
{: .fs-6 .fw-300 }

[View Projects on GitHub](https://github.com/marcus888-techstack/introduction-to-opencv/tree/main/projects){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Project Overview

```
                        6 Practical Projects
    ┌────────────────────────────────────────────────────────────┐
    │                   BEGINNER PROJECTS                        │
    ├────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────┐  ┌─────────────────────┐         │
    │  │   01 Document       │  │   02 Face           │         │
    │  │      Scanner        │  │      Attendance     │         │
    │  │  ┌───────────────┐  │  │  ┌───────────────┐  │         │
    │  │  │ Edge Detect   │  │  │  │ Face Detect   │  │         │
    │  │  │ Perspective   │  │  │  │ Recognition   │  │         │
    │  │  │ OCR Extract   │  │  │  │ Attendance    │  │         │
    │  │  └───────────────┘  │  │  └───────────────┘  │         │
    │  └─────────────────────┘  └─────────────────────┘         │
    ├────────────────────────────────────────────────────────────┤
    │                 INTERMEDIATE PROJECTS                      │
    ├────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────┐  ┌─────────────────────┐         │
    │  │   03 License        │  │   04 Object         │         │
    │  │      Plate          │  │      Counting       │         │
    │  │  ┌───────────────┐  │  │  ┌───────────────┐  │         │
    │  │  │ Plate Detect  │  │  │  │ Detection     │  │         │
    │  │  │ Character OCR │  │  │  │ Tracking      │  │         │
    │  │  │ Database      │  │  │  │ Analytics     │  │         │
    │  │  └───────────────┘  │  │  └───────────────┘  │         │
    │  └─────────────────────┘  └─────────────────────┘         │
    ├────────────────────────────────────────────────────────────┤
    │                   ADVANCED PROJECTS                        │
    ├────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────┐  ┌─────────────────────┐         │
    │  │   05 Quality        │  │   06 Gesture        │         │
    │  │      Inspection     │  │      Control        │         │
    │  │  ┌───────────────┐  │  │  ┌───────────────┐  │         │
    │  │  │ Defect Detect │  │  │  │ Hand Detect   │  │         │
    │  │  │ Classification│  │  │  │ Gesture Track │  │         │
    │  │  │ Reporting     │  │  │  │ App Control   │  │         │
    │  │  └───────────────┘  │  │  └───────────────┘  │         │
    │  └─────────────────────┘  └─────────────────────┘         │
    └────────────────────────────────────────────────────────────┘
```

---

## Beginner Projects (Sessions 1-2)

| Project | Description | Key Skills | README |
|:--------|:------------|:-----------|:-------|
| **01: Document Scanner** | Scan documents using webcam, apply perspective correction, extract text | Edge detection, contour finding, perspective transform, OCR | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/01_document_scanner/README.md) |
| **02: Face Attendance** | Automated attendance system using face recognition | Face detection, LBPH recognition, database management | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/02_face_attendance/README.md) |

---

## Intermediate Projects (Sessions 3-4)

| Project | Description | Key Skills | README |
|:--------|:------------|:-----------|:-------|
| **03: License Plate Recognition** | ANPR system for parking/security applications | Plate detection, character segmentation, OCR | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/03_license_plate/README.md) |
| **04: Object Counting** | Count and track people/vehicles with analytics | Object detection, tracking algorithms, data visualization | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/04_object_counting/README.md) |

---

## Advanced Projects (Sessions 5-6)

| Project | Description | Key Skills | README |
|:--------|:------------|:-----------|:-------|
| **05: Quality Inspection** | Industrial defect detection system | Image comparison, anomaly detection, classification | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/05_quality_inspection/README.md) |
| **06: Gesture Control** | Touchless presentation control using hand gestures | Hand detection, gesture recognition, system integration | [View README](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/projects/06_gesture_control/README.md) |

---

## Running Projects

Each project is self-contained with its own README and source code:

```bash
# Navigate to a project
cd projects/01_document_scanner

# Read the README for setup instructions
cat README.md

# Run the project
python main.py
```

---

## Project Structure

Each project folder contains:

```
projects/
├── 01_document_scanner/
│   ├── README.md          # Project documentation
│   ├── main.py            # Main application
│   ├── utils/             # Helper functions
│   └── samples/           # Sample images
├── 02_face_attendance/
│   └── ...
└── ...
```

---

## Prerequisites

Before starting projects, complete the relevant curriculum modules:

| Project | Required Modules |
|:--------|:-----------------|
| Document Scanner | Core, ImgProc, Calib3D |
| Face Attendance | Core, ObjDetect, Extras (Face) |
| License Plate | ImgProc, ObjDetect, Extras (OCR) |
| Object Counting | Video, ObjDetect, Extras (Tracking) |
| Quality Inspection | ImgProc, Features2D, ML |
| Gesture Control | Video, DNN, Extras |
