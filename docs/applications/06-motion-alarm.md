---
layout: default
title: "06: Motion Alarm"
parent: Applications
nav_order: 6
permalink: /applications/06-motion-alarm
---

# Motion Detection Alarm
{: .fs-9 }

Security camera-style motion detection system.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/06_motion_alarm.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Detect motion in video feed and trigger alerts. Perfect for security camera applications and home monitoring.

**Key Techniques:**
- Background subtraction
- Frame differencing
- Contour analysis
- Alert triggering

---

## How It Works

```
Frame → Background Model → Foreground Mask → Find Motion → Alert
   ↓          ↓                 ↓               ↓          ↓
[Current]  [Learned      [Moving         [Bounding   [Alarm!]
            background]   pixels]         boxes]
```

---

## Key OpenCV Functions

```python
# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

# Apply to frame
fg_mask = bg_subtractor.apply(frame)

# Remove shadows (gray pixels)
_, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

# Clean mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

# Find moving objects
contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > min_area:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Trigger alert!
```

---

## Background Subtractors

| Method | Speed | Quality | Best For |
|:-------|:------|:--------|:---------|
| MOG2 | Fast | Good | General use |
| KNN | Medium | Better | Varying lighting |
| Frame diff | Very fast | Basic | Simple scenes |

---

## Simple Frame Differencing

```python
# Alternative: Simple difference between frames
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

# Compute difference
diff = cv2.absdiff(prev_gray, curr_gray)
_, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
```

---

## Controls

| Key | Action |
|:----|:-------|
| `+/-` | Adjust sensitivity |
| `a` | Toggle alarm |
| `r` | Reset background |
| `s` | Save screenshot |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/06_motion_alarm.py
```

---

## Official Documentation

- [Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- [MOG2](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
