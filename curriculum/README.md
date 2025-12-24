# OpenCV Comprehensive Curriculum

A structured learning path covering all major OpenCV modules with **20+ hands-on tutorials**.

## Quick Start

```bash
# Run any module
python curriculum/01_core/01_basics.py
python curriculum/02_imgproc/01_filtering.py
```

---

## Main Modules (11 Modules, 17 Tutorials)

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

### Module 7: Camera Calibration (`07_calib3d/`)
Camera calibration and 3D geometry.
- **`01_camera_calibration.py`** - Calibration, undistortion, perspective transform

### Module 8: Deep Learning (`08_dnn/`)
Using neural networks in OpenCV.
- **`01_dnn_basics.py`** - Loading models, blob preparation, inference

### Module 9: Machine Learning (`09_ml/`)
Traditional ML with OpenCV.
- **`01_ml_basics.py`** - KNN, SVM, K-Means, Decision Trees

### Module 10: Computational Photography (`10_photo/`)
Image enhancement techniques.
- **`01_photo_basics.py`** - Inpainting, denoising, HDR, seamless cloning, stylization

### Module 11: Image Stitching (`11_stitching/`)
Creating panoramas.
- **`01_panorama.py`** - Stitcher API, manual pipeline, blending

---

## Extra Modules (`extras/`) - opencv-contrib

- **`01_face_module.py`** - Face recognition (Eigenfaces, Fisherfaces, LBPH)
- **`02_tracking.py`** - Object trackers (KCF, CSRT, MIL, etc.)
- **`03_text_ocr.py`** - Text detection (MSER, EAST) and OCR integration

---

## Module to Official Docs Mapping

| Module | OpenCV Docs Section |
|--------|---------------------|
| Core | [Core Operations](https://docs.opencv.org/4.x/d7/d16/tutorial_py_table_of_contents_core.html) |
| ImgProc | [Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html) |
| I/O & GUI | [GUI Features](https://docs.opencv.org/4.x/dc/d4d/tutorial_py_table_of_contents_gui.html) |
| Features2D | [Feature Detection](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html) |
| ObjDetect | [Object Detection](https://docs.opencv.org/4.x/d2/d64/tutorial_table_of_content_objdetect.html) |
| Video | [Video Analysis](https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html) |
| Calib3D | [Camera Calibration](https://docs.opencv.org/4.x/d9/db7/tutorial_py_table_of_contents_calib3d.html) |
| DNN | [Deep Learning](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html) |
| ML | [Machine Learning](https://docs.opencv.org/4.x/d6/de2/tutorial_py_table_of_contents_ml.html) |
| Photo | [Computational Photography](https://docs.opencv.org/4.x/d0/d25/tutorial_table_of_content_photo.html) |
| Stitching | [Image Stitching](https://docs.opencv.org/4.x/d1/d46/group__stitching.html) |

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

## Key Functions Per Module

| Module | Key Functions |
|--------|---------------|
| Core | `np.zeros`, `cv2.add`, `cv2.bitwise_and`, `cv2.split` |
| ImgProc | `cv2.blur`, `cv2.Canny`, `cv2.findContours`, `cv2.cvtColor` |
| I/O | `cv2.imread`, `cv2.VideoCapture`, `cv2.imshow` |
| Features | `cv2.ORB_create`, `cv2.BFMatcher`, `cv2.findHomography` |
| ObjDetect | `CascadeClassifier`, `cv2.matchTemplate` |
| Video | `cv2.calcOpticalFlowPyrLK`, `createBackgroundSubtractorMOG2` |
| Calib3D | `cv2.calibrateCamera`, `cv2.getPerspectiveTransform` |
| DNN | `cv2.dnn.readNet`, `cv2.dnn.blobFromImage` |
| ML | `cv2.ml.KNearest_create`, `cv2.ml.SVM_create`, `cv2.kmeans` |
| Photo | `cv2.inpaint`, `cv2.fastNlMeansDenoisingColored` |
| Stitching | `cv2.Stitcher_create`, `stitcher.stitch` |

---

## Prerequisites

```bash
pip install opencv-python numpy

# For extra modules
pip install opencv-contrib-python

# For OCR examples
pip install pytesseract easyocr
```
