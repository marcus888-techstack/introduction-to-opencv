---
layout: default
title: "02: Color Tracker"
parent: Applications
nav_order: 2
permalink: /applications/02-color-tracker
---

# Color Object Tracker
{: .fs-9 }

Track colored objects in real-time using HSV color space.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/02_color_tracker.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Track objects by their color in real-time video. Perfect for robotics, games, and interactive applications.

**Key Techniques:**
- HSV color space conversion
- Color range thresholding
- Contour detection
- Centroid tracking

---

## How It Works

```
Frame → HSV Convert → Color Mask → Find Contours → Track Center
   ↓         ↓            ↓              ↓             ↓
[BGR]    [H,S,V]      [Binary]     [Largest]     [x, y]
```

### Why HSV?

HSV (Hue, Saturation, Value) separates color from brightness:
- **Hue**: The actual color (0-180 in OpenCV)
- **Saturation**: Color intensity (0-255)
- **Value**: Brightness (0-255)

This makes color detection robust to lighting changes.

---

## Key OpenCV Functions

```python
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Create color mask
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Clean up mask
mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=2)

# Find largest contour
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

# Get center using moments
M = cv2.moments(largest)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
```

---

## Default Color Ranges

| Color | H Low | H High | S Range | V Range |
|:------|:------|:-------|:--------|:--------|
| Red | 0-10, 170-180 | - | 100-255 | 100-255 |
| Green | 35 | 85 | 100-255 | 100-255 |
| Blue | 100 | 130 | 100-255 | 100-255 |
| Yellow | 20 | 35 | 100-255 | 100-255 |

---

## Controls

| Key | Action |
|:----|:-------|
| `r` | Track red |
| `g` | Track green |
| `b` | Track blue |
| `y` | Track yellow |
| `+/-` | Adjust threshold |
| `s` | Save screenshot |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/02_color_tracker.py
```

---

## Official Documentation

- [Color Space Conversions](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- [Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
