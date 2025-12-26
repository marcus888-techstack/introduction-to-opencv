---
layout: default
title: "01: Document Scanner"
parent: Applications
nav_order: 1
permalink: /applications/01-document-scanner
---

# Document Scanner
{: .fs-9 }

Scan documents using edge detection and perspective transformation.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/01_document_scanner.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Transform photos of documents into clean, flat scans - like a mobile scanning app.

**Key Techniques:**
- Canny edge detection
- Contour detection and approximation
- Perspective transformation
- Adaptive thresholding

---

## How It Works

```
Input Image → Edge Detection → Find Document → Perspective Warp → Output
     ↓              ↓               ↓                ↓
 [Photo of    [Canny edges]  [4-corner        [Flattened
  document]                   contour]         document]
```

### Pipeline Steps

1. **Preprocessing**: Convert to grayscale, apply Gaussian blur
2. **Edge Detection**: Canny edge detector finds edges
3. **Contour Finding**: Find largest 4-sided contour (the document)
4. **Corner Ordering**: Sort corners to top-left, top-right, bottom-right, bottom-left
5. **Perspective Transform**: Warp to bird's-eye view
6. **Enhancement**: Adaptive threshold for clean text

---

## Key OpenCV Functions

```python
# Edge detection
edges = cv2.Canny(blur, 50, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Approximate to polygon
approx = cv2.approxPolyDP(contour, epsilon, True)

# Perspective transform
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(image, M, (width, height))

# Enhance text
result = cv2.adaptiveThreshold(warped, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```

---

## Controls

| Key | Action |
|:----|:-------|
| `c` | Capture and process document |
| `t` | Toggle threshold enhancement |
| `s` | Save scanned document |
| `r` | Reset |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/01_document_scanner.py
```

---

## Official Documentation

- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [Perspective Transform](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
