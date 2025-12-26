# Sample Data

Real images and videos for OpenCV tutorials.

## Quick Start

```bash
# Download all sample files
python curriculum/sample_data/download_samples.py

# Check status
python curriculum/sample_data/download_samples.py --check

# List available files
python curriculum/sample_data/download_samples.py --list
```

## Usage in Code

```python
import sys
sys.path.insert(0, 'curriculum')
from sample_data import get_image, get_video, quick_load

# Load specific image
img = get_image("lena.jpg")
gray = get_image("lena.jpg", cv2.IMREAD_GRAYSCALE)

# Quick load by category
face = quick_load("face")
fruits = quick_load("fruits")

# Get video path
cap = cv2.VideoCapture(get_video("vtest.avi"))
```

## Available Samples

### Images

| File | Description | Best For |
|------|-------------|----------|
| `lena.jpg` | Classic test image (face) | Face detection, filtering |
| `fruits.jpg` | Colorful fruits | Color segmentation, histograms |
| `baboon.jpg` | Detailed texture | Filtering, sharpening |
| `building.jpg` | Architecture | Edge detection, lines |
| `sudoku.png` | Sudoku puzzle | Perspective transform, OCR |
| `box.png` | Simple box | Feature matching (query) |
| `box_in_scene.png` | Box in scene | Feature matching (scene) |
| `chessboard.png` | Calibration pattern | Camera calibration |
| `imageTextN.png` | Text sample | OCR, text detection |
| `j.png` | Letter J | Morphological operations |
| `coins.jpg` | Coins | Object counting, watershed |

### Videos

| File | Description | Best For |
|------|-------------|----------|
| `vtest.avi` | Walking people | Motion detection, tracking |
| `slow_traffic_small.mp4` | Traffic scene | Vehicle detection |

## Sources

- [OpenCV GitHub Samples](https://github.com/opencv/opencv/tree/4.x/samples/data)
- [scikit-image Data](https://github.com/scikit-image/scikit-image/tree/main/skimage/data)

## Adding Custom Samples

Place your own images in this folder and load them:

```python
import cv2
import os

sample_dir = os.path.dirname(__file__)
img = cv2.imread(os.path.join(sample_dir, "my_image.jpg"))
```
