---
layout: default
title: Getting Started
nav_order: 2
description: "Installation and setup guide for the OpenCV course"
permalink: /getting-started
---

# Getting Started
{: .fs-9 }

Set up your development environment and run your first OpenCV program.
{: .fs-6 .fw-300 }

---

## Prerequisites

Before starting this course, ensure you have:

- **Python 3.8 or higher** installed
- **pip** package manager
- **Webcam** (recommended for real-time projects)
- Basic familiarity with Python and NumPy

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/marcus888-techstack/introduction-to-opencv.git
cd introduction-to-opencv
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` - Main OpenCV package
- `opencv-contrib-python` - Extra modules (face, tracking, etc.)
- `numpy` - Array operations
- `pytesseract` - OCR integration
- `easyocr` - Alternative OCR

### Step 4: Verify Installation

```python
import cv2
print(f"OpenCV version: {cv2.__version__}")
```

Expected output: `OpenCV version: 4.x.x`

---

## Repository Structure

```
introduction-to-opencv/
├── curriculum/              # Learning modules (11 + extras)
│   ├── 01_core/            # Core operations
│   ├── 02_imgproc/         # Image processing
│   ├── 03_io_gui/          # I/O and GUI
│   ├── 04_features2d/      # Feature detection
│   ├── 05_objdetect/       # Object detection
│   ├── 06_video/           # Video analysis
│   ├── 07_calib3d/         # Camera calibration
│   ├── 08_dnn/             # Deep learning
│   ├── 09_ml/              # Machine learning
│   ├── 10_photo/           # Computational photography
│   ├── 11_stitching/       # Image stitching
│   └── extras/             # Face, tracking, OCR
│
├── projects/               # 6 practical projects
│   ├── 01_document_scanner/
│   ├── 02_face_attendance/
│   ├── 03_license_plate/
│   ├── 04_object_counting/
│   ├── 05_quality_inspection/
│   └── 06_gesture_control/
│
├── assets/                 # Sample images and data
├── utils/                  # Shared utilities
├── requirements.txt        # Dependencies
└── README.md
```

---

## Running Tutorials

Each curriculum module contains Python scripts you can run directly:

```bash
# Run Core basics tutorial
python curriculum/01_core/01_basics.py

# Run Image Processing filtering tutorial
python curriculum/02_imgproc/01_filtering.py

# Run Feature detection tutorial
python curriculum/04_features2d/01_corners.py
```

### Interactive Controls

Most tutorials have keyboard controls:

| Key | Action |
|:----|:-------|
| `q` or `ESC` | Quit the program |
| `s` | Save current image |
| `Space` | Pause/resume video |
| `+` / `-` | Adjust parameters |

---

## Running Projects

Projects are complete applications with a `main.py` entry point:

```bash
# Document Scanner
cd projects/01_document_scanner
python main.py

# Face Attendance System
cd projects/02_face_attendance
python main.py
```

---

## Sample Images

The `assets/` folder contains sample images for testing:

```bash
assets/
├── images/         # Sample images (lena.png, etc.)
├── videos/         # Sample video clips
└── models/         # Pre-trained model files
```

If you need additional test images, you can:

1. Use your own images
2. Use OpenCV's built-in samples:
   ```python
   import cv2
   img = cv2.imread(cv2.samples.findFile('lena.jpg'))
   ```

---

## Webcam Setup

Many tutorials use your webcam. Test it with:

```python
import cv2

cap = cv2.VideoCapture(0)  # 0 = default camera
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### Troubleshooting Webcam Issues

- **No camera found**: Try `VideoCapture(1)` or `VideoCapture(2)`
- **Permission denied (macOS)**: Grant camera access in System Preferences > Privacy
- **Virtual camera**: Install OBS Virtual Camera for testing

---

## Learning Path

Recommended order for completing the curriculum:

```
Week 1: Foundations
├── Day 1: 01_core + 03_io_gui (basics)
├── Day 2: 02_imgproc (filtering, morphology)
└── Day 3: 02_imgproc (contours, histograms)

Week 2: Features & Detection
├── Day 1: 04_features2d (detectors, matchers)
├── Day 2: 05_objdetect (Haar, templates)
└── Day 3: 06_video (tracking, optical flow)

Week 3: Advanced Topics
├── Day 1: 07_calib3d (calibration, 3D)
├── Day 2: 08_dnn (deep learning)
└── Day 3: 09_ml + 10_photo

Week 4: Projects
├── Day 1-2: Project implementation
└── Day 3: 11_stitching + extras
```

---

## Getting Help

- **Module README files**: Each curriculum folder has a README with algorithm explanations
- **Official Docs**: [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- **GitHub Issues**: Report bugs or ask questions in the repository

---

## Next Steps

Ready to start learning? Begin with the first module:

[Start with Core Operations](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/01_core/README.md){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }
[View All Modules]({{ site.baseurl }}/modules){: .btn .btn-outline .fs-5 .mb-4 .mb-md-0 }
