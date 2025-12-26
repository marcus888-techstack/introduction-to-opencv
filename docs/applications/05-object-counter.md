---
layout: default
title: "05: Object Counter"
parent: Applications
nav_order: 5
permalink: /applications/05-object-counter
---

# Object Counter
{: .fs-9 }

Count objects in images using contour detection.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/05_object_counter.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Automatically count objects like coins, pills, or items on a conveyor belt using image segmentation and contour analysis.

**Key Techniques:**
- Adaptive thresholding
- Morphological operations
- Contour detection
- Connected component analysis

---

## How It Works

```
Image → Grayscale → Threshold → Morphology → Find Contours → Count
   ↓        ↓           ↓           ↓             ↓           ↓
[RGB]   [Gray]     [Binary]    [Clean]      [Objects]    [N=15]
```

---

## Key OpenCV Functions

```python
# Convert and threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Clean up with morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter by size and count
min_area = 100
objects = [c for c in contours if cv2.contourArea(c) > min_area]
count = len(objects)
```

---

## Connected Components Alternative

```python
# Label connected regions
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

# stats contains: x, y, width, height, area for each component
for i in range(1, num_labels):  # Skip background (0)
    x, y, w, h, area = stats[i]
    if area > min_area:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

---

## Watershed for Touching Objects

When objects touch, use watershed segmentation:

```python
# Distance transform
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)

# Find markers
_, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))

# Apply watershed
markers = cv2.watershed(image, markers)
```

---

## Controls

| Key | Action |
|:----|:-------|
| `+/-` | Adjust threshold |
| `m` | Toggle morphology |
| `w` | Use watershed |
| `s` | Save result |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/05_object_counter.py
```

---

## Official Documentation

- [Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
