---
layout: default
title: "08: Lane Detection"
parent: Applications
nav_order: 8
permalink: /applications/08-lane-detection
---

# Lane Detection
{: .fs-9 }

Detect road lanes for autonomous driving applications.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/08_lane_detection.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Detect lane markings on roads using classical computer vision techniques. Foundation for self-driving car systems.

**Key Techniques:**
- Canny edge detection
- Region of interest masking
- Hough line transform
- Line averaging and extrapolation

---

## Pipeline

```
Frame → Grayscale → Blur → Canny → ROI Mask → Hough Lines → Average → Draw
   ↓        ↓        ↓       ↓         ↓           ↓          ↓       ↓
[Road]  [Gray]    [Smooth] [Edges]  [Trapezoid] [Lines]   [L/R]   [Overlay]
```

---

## Key OpenCV Functions

```python
# 1. Preprocess
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. Edge detection
edges = cv2.Canny(blur, 50, 150)

# 3. Region of interest (trapezoid on road)
mask = np.zeros_like(edges)
polygon = np.array([[
    (width * 0.1, height),
    (width * 0.45, height * 0.6),
    (width * 0.55, height * 0.6),
    (width * 0.9, height)
]], dtype=np.int32)
cv2.fillPoly(mask, polygon, 255)
masked = cv2.bitwise_and(edges, mask)

# 4. Hough lines
lines = cv2.HoughLinesP(
    masked,
    rho=2,
    theta=np.pi/180,
    threshold=50,
    minLineLength=40,
    maxLineGap=100
)
```

---

## Line Averaging

```python
def average_slope_intercept(lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < 0:  # Left lane (negative slope)
            left_fit.append((slope, intercept))
        else:          # Right lane (positive slope)
            right_fit.append((slope, intercept))

    # Average the lines
    left_avg = np.average(left_fit, axis=0)
    right_avg = np.average(right_fit, axis=0)

    return left_avg, right_avg
```

---

## Hough Transform Parameters

| Parameter | Purpose | Typical Value |
|:----------|:--------|:--------------|
| `rho` | Distance resolution | 1-2 pixels |
| `theta` | Angle resolution | π/180 radians |
| `threshold` | Minimum votes | 50-100 |
| `minLineLength` | Shortest line | 40-100 pixels |
| `maxLineGap` | Gap tolerance | 50-100 pixels |

---

## Controls

| Key | Action |
|:----|:-------|
| `+/-` | Adjust Canny thresholds |
| `s` | Save screenshot |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/08_lane_detection.py
```

---

## Official Documentation

- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Hough Line Transform](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html)
