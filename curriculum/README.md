# OpenCV Comprehensive Curriculum

A structured learning path covering all major OpenCV modules with **20+ hands-on tutorials**.

## Quick Start

```bash
# Run any module
python curriculum/01_core/01_basics.py
python curriculum/02_imgproc/01_filtering.py
```

---

## Main Modules (11 Modules, 30+ Tutorials)

### Module 1: Core Functionality (`01_core/`)
Foundation of OpenCV - arrays, basic operations, and pixel manipulation.
- **`01_basics.py`** - Mat, pixel access, arithmetic, bitwise ops, channels, ROI, borders

### Module 2: Image Processing (`02_imgproc/`)
Image filtering, transformations, and analysis.
- **`01_filtering.py`** - Blur, Gaussian, median, bilateral, custom kernels, sharpening
- **`02_morphology.py`** - Erosion, dilation, opening, closing, gradient, top/black hat
- **`03_edges_contours.py`** - Sobel, Laplacian, Canny, contours, shape detection
- **`04_color_histogram.py`** - Color spaces, HSV, histograms, CLAHE, color segmentation

### Module 3: I/O and GUI (`03_io_gui/`)
Reading, writing, and displaying images/videos.
- **`01_image_io.py`** - imread, imwrite, formats, encoding
- **`02_video_io.py`** - VideoCapture, VideoWriter, camera input
- **`03_gui_basics.py`** - Windows, keyboard, trackbars, mouse events, drawing

### Module 4: Features2D (`04_features2d/`)
Feature detection, description, and matching.
- **`01_corners.py`** - Harris, Shi-Tomasi, FAST, subpixel accuracy
- **`02_descriptors.py`** - ORB, SIFT, BRISK, AKAZE
- **`03_matching.py`** - BF matcher, FLANN, ratio test, homography

### Module 5: Object Detection (`05_objdetect/`)
Object detection methods.
- **`01_cascade_classifiers.py`** - Haar cascades, face/eye detection
- **`02_template_matching.py`** - Template matching, multi-scale, NMS

### Module 6: Video Analysis (`06_video/`)
Motion analysis and background modeling.
- **`01_optical_flow.py`** - Lucas-Kanade, Farneback, motion detection
- **`02_background_subtraction.py`** - MOG2, KNN, foreground detection

### Module 7: Machine Learning (`07_ml/`)
Traditional ML with OpenCV.
- **`01_ml_basics.py`** - KNN, SVM, K-Means, Decision Trees fundamentals
- **`02_digit_recognition.py`** - Handwritten digit classification with real data
- **`03_hog_svm.py`** - HOG features + SVM for pedestrian detection
- **`04_kmeans_segmentation.py`** - Image segmentation using K-Means clustering

### Module 8: Deep Learning (`08_dnn/`)
Using neural networks in OpenCV.
- **`01_dnn_basics.py`** - Loading models, blob preparation, inference
- **`02_dnn_video.py`** - Real-time video inference
- **`03_dnn_formats.py`** - ONNX, TensorFlow, Caffe model formats

### Module 9: Multi-Object Tracking (`09_mcmot/`)
Multi-camera multi-object tracking with Re-ID.
- **`01_tracking_basics.py`** - OpenCV tracking API, single object trackers
- **`02_person_detection.py`** - YOLOv4-tiny person detection
- **`03_person_reid.py`** - Person re-identification with deep features
- **`04_mot_tracker.py`** - Multi-object tracking with SORT/DeepSORT concepts
- **`05_mcmot_multicam.py`** - Cross-camera tracking and Re-ID matching

### Module 10: Camera Calibration & 3D Vision (`10_calib3d/`)
Camera calibration and 3D geometry.
- **`01_camera_calibration.py`** - Calibration, undistortion, perspective transform
- **`02_pose_estimation.py`** - 3D pose estimation with PnP
- **`03_stereo_vision.py`** - Stereo calibration, rectification, disparity maps
- **`04_3d_reconstruction.py`** - Depth estimation and point cloud generation
- **`05_sfm_concepts.py`** - Structure from Motion fundamentals

### Module 11: Image Stitching (`11_stitching/`)
Creating panoramas with various techniques.
- **`01_panorama.py`** - High-level Stitcher API, basic manual stitch
- **`02_manual_stitching.py`** - Step-by-step pipeline: features, matching, RANSAC, warping
- **`03_blending_techniques.py`** - Blending comparison: none, alpha, feather, multi-band
- **`04_cylindrical_pano.py`** - Cylindrical/spherical projections, wide panoramas

---

## Extra Modules (`extras/`) - opencv-contrib

- **`01_face_module.py`** - Face recognition (Eigenfaces, Fisherfaces, LBPH)
- **`02_tracking.py`** - Object trackers (KCF, CSRT, MIL, etc.)
- **`03_text_ocr.py`** - Text detection (MSER, EAST) and OCR integration

---

## Official Documentation

**Main Reference:** [OpenCV 4.x Documentation](https://docs.opencv.org/4.x/)

### Module to Official Docs Mapping

| Module | OpenCV Docs Section |
|--------|---------------------|
| 01 Core | [Core Operations](https://docs.opencv.org/4.x/d7/d16/tutorial_py_table_of_contents_core.html) |
| 02 ImgProc | [Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html) |
| 03 I/O & GUI | [GUI Features](https://docs.opencv.org/4.x/dc/d4d/tutorial_py_table_of_contents_gui.html) |
| 04 Features2D | [Feature Detection](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html) |
| 05 ObjDetect | [Object Detection](https://docs.opencv.org/4.x/d2/d64/tutorial_table_of_content_objdetect.html) |
| 06 Video | [Video Analysis](https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html) |
| 07 ML | [Machine Learning](https://docs.opencv.org/4.x/d6/de2/tutorial_py_table_of_contents_ml.html) |
| 08 DNN | [Deep Learning](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html) |
| 09 MCMOT | [Video Analysis](https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html) + DNN |
| 10 Calib3D | [Camera Calibration](https://docs.opencv.org/4.x/d9/db7/tutorial_py_table_of_contents_calib3d.html) |
| 11 Stitching | [Image Stitching](https://docs.opencv.org/4.x/d1/d46/group__stitching.html) |

### Recommended Tutorials for Examples

Tutorials marked with visual output and clear code examples:

#### Image Processing (Best for Visual Demos)
| Tutorial | URL | Why Good for Examples |
|----------|-----|----------------------|
| Image Thresholding | [tutorial_py_thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) | Clear before/after visuals, Otsu's method |
| Canny Edge Detection | [tutorial_py_canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) | Iconic CV algorithm, dramatic visual results |
| Morphological Ops | [tutorial_py_morphological_ops](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) | Step-by-step erosion/dilation demos |
| Contours | [tutorial_py_contours](https://docs.opencv.org/4.x/d3/d05/tutorial_py_table_of_contents_contours.html) | Shape detection, hierarchy visualization |
| Hough Lines | [tutorial_py_houghlines](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html) | Line detection on real images |
| Hough Circles | [tutorial_py_houghcircles](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html) | Circle detection, coin counting example |
| Watershed Segmentation | [tutorial_py_watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) | Advanced segmentation with markers |
| GrabCut | [tutorial_py_grabcut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html) | Interactive foreground extraction |

#### Feature Detection (Good for Matching Demos)
| Tutorial | URL | Why Good for Examples |
|----------|-----|----------------------|
| Harris Corners | [tutorial_py_features_harris](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html) | Classic corner detection visualization |
| SIFT | [tutorial_py_sift_intro](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) | Scale-invariant features, keypoint visualization |
| ORB | [tutorial_py_orb](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html) | Fast, free alternative to SIFT/SURF |
| Feature Matching | [tutorial_py_matcher](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html) | BF/FLANN matching with draw functions |
| Homography | [tutorial_py_feature_homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) | Object detection via feature matching |

#### Video & Real-time (Good for Interactive Demos)
| Tutorial | URL | Why Good for Examples |
|----------|-----|----------------------|
| Color Tracking | [tutorial_py_colorspaces](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) | Real-time object tracking by color |
| Optical Flow | [tutorial_py_lucas_kanade](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) | Motion visualization, tracking points |
| Background Subtraction | [tutorial_py_bg_subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html) | MOG2/KNN foreground detection |

#### 3D & Calibration (Good for Understanding Geometry)
| Tutorial | URL | Why Good for Examples |
|----------|-----|----------------------|
| Camera Calibration | [tutorial_py_calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) | Chessboard detection, undistortion |
| Depth Map | [tutorial_py_depthmap](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html) | Stereo vision depth estimation |

#### Deep Learning & ML
| Tutorial | URL | Why Good for Examples |
|----------|-----|----------------------|
| DNN Object Detection | [js_object_detection](https://docs.opencv.org/4.x/js_object_detection.html) | Real-time detection with pre-trained models |
| K-Means Clustering | [tutorial_py_kmeans](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html) | Color quantization, dominant colors |

---

## Learning Path

```
Week 1: Foundations
├── Day 1: 01_core + 03_io_gui (basics)
├── Day 2: 02_imgproc (filtering, morphology)
└── Day 3: 02_imgproc (contours, histograms)

Week 2: Features & Detection
├── Day 1: 04_features2d (detectors, matchers)
├── Day 2: 05_objdetect (Haar, templates)
└── Day 3: 06_video (optical flow, background subtraction)

Week 3: ML & Deep Learning
├── Day 1: 07_ml (KNN, SVM, K-Means, Decision Trees)
├── Day 2: 08_dnn (model loading, inference)
└── Day 3: 09_mcmot (tracking, Re-ID)

Week 4: 3D Vision & Panoramas
├── Day 1: 10_calib3d (calibration, stereo)
├── Day 2: 10_calib3d (3D reconstruction, SfM)
└── Day 3: 11_stitching (panoramas, blending)
```

---

## Key Functions Per Module

| Module | Key Functions |
|--------|---------------|
| 01 Core | `np.zeros`, `cv2.add`, `cv2.bitwise_and`, `cv2.split` |
| 02 ImgProc | `cv2.blur`, `cv2.Canny`, `cv2.findContours`, `cv2.cvtColor` |
| 03 I/O | `cv2.imread`, `cv2.VideoCapture`, `cv2.imshow` |
| 04 Features | `cv2.ORB_create`, `cv2.BFMatcher`, `cv2.findHomography` |
| 05 ObjDetect | `CascadeClassifier`, `cv2.matchTemplate` |
| 06 Video | `cv2.calcOpticalFlowPyrLK`, `createBackgroundSubtractorMOG2` |
| 07 ML | `cv2.ml.KNearest_create`, `cv2.ml.SVM_create`, `cv2.kmeans` |
| 08 DNN | `cv2.dnn.readNet`, `cv2.dnn.blobFromImage` |
| 09 MCMOT | `cv2.TrackerCSRT_create`, `cv2.dnn.readNet` (YOLO, Re-ID) |
| 10 Calib3D | `cv2.calibrateCamera`, `cv2.solvePnP`, `cv2.stereoCalibrate` |
| 11 Stitching | `cv2.Stitcher_create`, `cv2.findHomography`, `cv2.warpPerspective` |

---

## Sample Data

Download real images and videos for tutorials:

```bash
# Download all sample images/videos
python curriculum/sample_data/download_samples.py

# Check what's available
python curriculum/sample_data/download_samples.py --check
```

Use in code:
```python
from sample_data import get_image, get_video

img = get_image("lena.jpg")
cap = cv2.VideoCapture(get_video("vtest.avi"))
```

---

## Practical Applications (`applications/`)

Real-world projects combining multiple techniques:

### Beginner
| App | Description | Key Techniques |
|-----|-------------|----------------|
| `01_document_scanner.py` | Mobile scanner app | Edges, perspective transform |
| `02_color_tracker.py` | Track colored objects | HSV, contours |
| `03_realtime_filters.py` | Instagram-style filters | Custom kernels, LUTs |
| `04_face_blur.py` | Privacy protection | Face detection, blur |
| `05_object_counter.py` | Count objects | Contours, watershed |

### Intermediate
| App | Description | Key Techniques |
|-----|-------------|----------------|
| `06_motion_alarm.py` | Security camera | Background subtraction |

*(More applications coming: QR reader, lane detection, watermarking, etc.)*

---

## Prerequisites

```bash
pip install opencv-python numpy

# For extra modules
pip install opencv-contrib-python

# For OCR examples
pip install pytesseract easyocr
```
