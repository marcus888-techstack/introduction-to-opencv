---
layout: default
title: "04: Face Blur"
parent: Applications
nav_order: 4
permalink: /applications/04-face-blur
---

# Face Blur Privacy
{: .fs-9 }

Automatically detect and blur faces for privacy protection.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/04_face_blur.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Detect faces in images or video and apply blur for privacy protection. Useful for social media, journalism, and security applications.

**Key Techniques:**
- Haar cascade face detection
- Gaussian blur
- Pixelation effect
- Region of interest processing

---

## How It Works

```
Frame → Face Detection → Extract ROI → Apply Blur → Replace ROI
   ↓          ↓              ↓            ↓            ↓
[Image]  [Bounding      [Face        [Blurred     [Final
          boxes]         region]      face]        image]
```

---

## Key OpenCV Functions

```python
# Load cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Detect faces
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Blur each face
for (x, y, w, h) in faces:
    roi = frame[y:y+h, x:x+w]

    # Gaussian blur
    blurred = cv2.GaussianBlur(roi, (99, 99), 30)

    # Or pixelation
    small = cv2.resize(roi, (10, 10))
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    frame[y:y+h, x:x+w] = blurred  # or pixelated
```

---

## Blur Types

| Type | Method | Use Case |
|:-----|:-------|:---------|
| Gaussian | `GaussianBlur` | Natural soft blur |
| Pixelation | Resize down then up | Mosaic/censored look |
| Box | `blur` | Simple average |
| Median | `medianBlur` | Preserves edges |

---

## Detection Parameters

```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,    # Image pyramid scale (smaller = more accurate, slower)
    minNeighbors=5,     # Higher = fewer false positives
    minSize=(30, 30)    # Minimum face size to detect
)
```

---

## Controls

| Key | Action |
|:----|:-------|
| `g` | Gaussian blur |
| `p` | Pixelation |
| `+/-` | Adjust blur strength |
| `s` | Save screenshot |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/04_face_blur.py
```

---

## Official Documentation

- [Face Detection](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [Image Smoothing](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
