# OpenCV Practical Applications

Real-world projects combining multiple OpenCV techniques.

## Quick Start

```bash
# Download sample images/videos first
python curriculum/sample_data/download_samples.py

# Run any application
python curriculum/applications/01_document_scanner.py
python curriculum/applications/02_color_tracker.py
```

---

## Beginner Applications

| # | Application | Key Techniques | Use Case |
|---|-------------|----------------|----------|
| 01 | [Document Scanner](01_document_scanner.py) | Edge detection, Perspective transform, Thresholding | Mobile scanning apps |
| 02 | [Color Object Tracker](02_color_tracker.py) | HSV color space, Contours, Video | Robotics, interactive games |
| 03 | [Real-time Filters](03_realtime_filters.py) | Custom kernels, Blending, LUTs | Instagram/TikTok filters |
| 04 | [Face Blur Privacy](04_face_blur.py) | Cascade classifier, Gaussian blur | Privacy protection |
| 05 | [Object Counter](05_object_counter.py) | Thresholding, Contours, Connected components | Inventory counting |

## Intermediate Applications

| # | Application | Key Techniques | Use Case |
|---|-------------|----------------|----------|
| 06 | [Motion Detection Alarm](06_motion_alarm.py) | Background subtraction, Bounding boxes | Security cameras |
| 07 | [QR/Barcode Reader](07_qr_barcode_reader.py) | QRCodeDetector, BarcodeDetector | Payments, inventory |
| 08 | [Lane Detection](08_lane_detection.py) | Canny edges, Hough lines, ROI masking | Self-driving cars |
| 09 | [Image Watermarking](09_image_watermark.py) | Alpha blending, LSB steganography | Copyright protection |
| 10 | [Color Palette Extractor](10_color_palette_extractor.py) | K-means clustering, Color quantization | Design tools |

## Advanced Applications

| # | Application | Key Techniques | Use Case |
|---|-------------|----------------|----------|
| 11 | [ArUco Marker Detection](11_aruco_detection.py) | ArUco dictionary, Pose estimation | Augmented reality |
| 12 | [Hand Gesture Recognition](12_hand_gesture.py) | Skin segmentation, Convex hull, Defects | Gesture control |
| 13 | [Virtual Background](13_virtual_background.py) | Background subtraction, GrabCut, Color keying | Video conferencing |
| 14 | [Panorama Stitcher](14_panorama_stitcher.py) | Feature matching, Homography, Blending | Photography apps |

---

## Techniques Matrix

| Application | ImgProc | Features | ObjDetect | Video | ML |
|-------------|---------|----------|-----------|-------|-----|
| Document Scanner | X | | | | |
| Color Tracker | X | | | X | |
| Real-time Filters | X | | | X | |
| Face Blur | | | X | X | |
| Object Counter | X | | | | |
| Motion Alarm | | | | X | |
| QR Reader | | | X | X | |
| Lane Detection | X | | | X | |
| Watermarking | X | | | | |
| Color Extractor | X | | | | X |
| ArUco Markers | | X | X | X | |
| Hand Gesture | X | | | X | |
| Virtual Background | | | | X | |
| Panorama | | X | | | |

---

## Application Details

### Beginner (01-05)
These applications use fundamental OpenCV techniques and are great starting points:
- Basic image processing (filtering, thresholding, morphology)
- Contour detection and analysis
- Color space conversions
- Simple video capture

### Intermediate (06-10)
These build on the basics with more advanced techniques:
- Background subtraction for motion detection
- Built-in detectors (QR codes, barcodes)
- Classical computer vision algorithms (Hough transform)
- Clustering algorithms (K-Means)

### Advanced (11-14)
These combine multiple techniques for sophisticated applications:
- Marker-based AR with pose estimation
- Gesture recognition using convexity analysis
- Real-time video segmentation
- Multi-image feature matching and homography

---

## Prerequisites

```bash
# Core OpenCV
pip install opencv-python numpy

# For advanced applications (ArUco, SIFT)
pip install opencv-contrib-python
```

## Sample Data

Applications automatically try to use real sample images. To download them:

```bash
python curriculum/sample_data/download_samples.py
```

Without sample images, applications fall back to:
1. Webcam input (if available)
2. Synthetic demo images

---

## Running Applications

Each application has two modes:

1. **Interactive Mode** (default): Uses webcam with keyboard controls
2. **Demo Mode**: Falls back to static images if webcam unavailable

```bash
# Interactive mode (requires webcam)
python curriculum/applications/02_color_tracker.py

# Most apps show controls on startup:
# - Press 'q' to quit
# - Press 's' to save screenshot
# - Various keys for options
```

## Creating Your Own Application

Use this template structure:

```python
"""
Application XX: Your App Name
=============================
Brief description.

Techniques Used:
- Technique 1
- Technique 2

Official Docs:
- https://docs.opencv.org/4.x/...
"""

import cv2
import numpy as np
import sys
import os

# Add parent for sample_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image, get_video


class YourProcessor:
    """Main processing class."""

    def __init__(self):
        pass

    def process(self, frame):
        # Your processing logic
        return frame


def load_demo_image():
    """Load real image or create fallback."""
    for sample in ["image1.jpg", "image2.jpg"]:
        img = get_image(sample)
        if img is not None:
            return img
    # Synthetic fallback
    return np.zeros((480, 640, 3), dtype=np.uint8)


def interactive_mode():
    """Webcam-based interactive mode."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        demo_mode()
        return

    processor = YourProcessor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = processor.process(frame)
        cv2.imshow("Result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """Static image demo."""
    img = load_demo_image()
    processor = YourProcessor()
    result = processor.process(img)

    cv2.imshow("Demo", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application XX: Your App Name")
    print("=" * 60)

    try:
        interactive_mode()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
```
