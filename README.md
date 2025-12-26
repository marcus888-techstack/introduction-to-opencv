# Introduction to OpenCV

A comprehensive OpenCV curriculum with **11 modules, 30+ tutorials**, and real-world applications.

## Curriculum Overview

| Module | Topics | Tutorials |
|--------|--------|-----------|
| **01 Core** | Image basics, NumPy arrays, pixel operations | 1 |
| **02 Image Processing** | Filtering, morphology, edges, contours, histograms | 4 |
| **03 I/O & GUI** | Image/video I/O, windows, trackbars, drawing | 3 |
| **04 Features2D** | Corner detection, SIFT, ORB, feature matching | 3 |
| **05 Object Detection** | Haar cascades, template matching | 2 |
| **06 Video Analysis** | Optical flow, background subtraction | 2 |
| **07 Machine Learning** | KNN, SVM, K-Means, HOG+SVM | 4 |
| **08 Deep Learning** | DNN module, model loading, inference | 3 |
| **09 Multi-Object Tracking** | MOT, MCMOT, person Re-ID | 5 |
| **10 3D Vision** | Camera calibration, stereo, SfM | 5 |
| **11 Image Stitching** | Panoramas, homography, blending | 4 |

## Quick Start

```bash
# Install dependencies
pip install opencv-python opencv-contrib-python numpy

# Download sample images/videos
python curriculum/sample_data/download_samples.py

# Run any tutorial
python curriculum/01_core/01_basics.py
python curriculum/02_imgproc/01_filtering.py
```

## Documentation

- **[Online Docs](https://marcus888-techstack.github.io/introduction-to-opencv/)** - GitHub Pages documentation
- **[Teaching Materials](https://marcus888-techstack.github.io/introduction-to-opencv/teaching-materials)** - PDF guides for each module
- **[Curriculum Details](curriculum/README.md)** - Full module breakdown

## Learning Path

```
Week 1: Foundations
├── Day 1: Core + I/O & GUI
├── Day 2: Image Processing (filtering, morphology)
└── Day 3: Image Processing (contours, histograms)

Week 2: Features & Detection
├── Day 1: Features2D (detectors, matchers)
├── Day 2: Object Detection (Haar, templates)
└── Day 3: Video Analysis (optical flow, background subtraction)

Week 3: ML & Deep Learning
├── Day 1: Machine Learning (KNN, SVM, K-Means)
├── Day 2: Deep Learning (DNN module)
└── Day 3: Multi-Object Tracking (MOT, Re-ID)

Week 4: 3D Vision & Panoramas
├── Day 1: Camera Calibration & Stereo
├── Day 2: 3D Reconstruction & SfM
└── Day 3: Image Stitching & Panoramas
```

## Practical Applications

Real-world projects combining multiple techniques:

| Application | Description |
|-------------|-------------|
| Document Scanner | Edge detection, perspective transform |
| Color Tracker | Real-time object tracking by color |
| Face Blur | Privacy protection with face detection |
| Motion Alarm | Security camera with background subtraction |
| QR/Barcode Reader | Decode QR codes and barcodes |
| Lane Detection | Road lane detection for ADAS |
| ArUco Detection | Augmented reality markers |
| Panorama Stitcher | Create wide panoramas |

See `curriculum/applications/` for full implementations.

## Official OpenCV Documentation

- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)
- [Feature Detection](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [Video Analysis](https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html)
- [Machine Learning](https://docs.opencv.org/4.x/d6/de2/tutorial_py_table_of_contents_ml.html)
- [Deep Learning](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)

## Requirements

- Python 3.8+
- OpenCV 4.x
- NumPy
- Webcam (for real-time demos)

```bash
pip install -r requirements.txt
```
