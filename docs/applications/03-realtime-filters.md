---
layout: default
title: "03: Real-time Filters"
parent: Applications
nav_order: 3
permalink: /applications/03-realtime-filters
---

# Real-time Filters
{: .fs-9 }

Apply Instagram/TikTok-style filters to live video.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/03_realtime_filters.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Apply various visual filters to webcam feed in real-time, similar to social media apps.

**Key Techniques:**
- Custom convolution kernels
- Color manipulation
- Image blending
- Look-up tables (LUTs)

---

## Available Filters

| Filter | Technique | Effect |
|:-------|:----------|:-------|
| Grayscale | Color conversion | Black and white |
| Sepia | Matrix multiplication | Vintage brown tone |
| Negative | Inversion | Inverted colors |
| Sketch | Edge detection + threshold | Pencil drawing |
| Emboss | Custom kernel | 3D relief effect |
| Sharpen | Custom kernel | Enhanced edges |
| Blur | Gaussian filter | Soft focus |
| Warm | Channel adjustment | Orange/yellow tint |
| Cool | Channel adjustment | Blue tint |
| Vignette | Gradient mask | Dark corners |

---

## Key OpenCV Functions

```python
# Grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Sepia (matrix multiplication)
sepia_kernel = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])
sepia = cv2.transform(frame, sepia_kernel)

# Custom kernels
emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
embossed = cv2.filter2D(frame, -1, emboss_kernel)

# Sketch effect
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
inv = 255 - gray
blur = cv2.GaussianBlur(inv, (21, 21), 0)
sketch = cv2.divide(gray, 255 - blur, scale=256)
```

---

## Vignette Effect

```python
def create_vignette(shape, strength=0.5):
    rows, cols = shape[:2]
    X = cv2.getGaussianKernel(cols, cols * strength)
    Y = cv2.getGaussianKernel(rows, rows * strength)
    mask = Y * X.T
    mask = mask / mask.max()
    return mask
```

---

## Controls

| Key | Action |
|:----|:-------|
| `1-9`, `0` | Select filter |
| `n` | Next filter |
| `p` | Previous filter |
| `s` | Save screenshot |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/03_realtime_filters.py
```

---

## Official Documentation

- [Image Filtering](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
- [Geometric Transforms](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
