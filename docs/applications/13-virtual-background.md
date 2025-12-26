---
layout: default
title: "13: Virtual Background"
parent: Applications
nav_order: 13
permalink: /applications/13-virtual-background
---

# Virtual Background
{: .fs-9 }

Replace video background like Zoom/Teams using segmentation.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/13_virtual_background.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Replace the background in real-time video, similar to virtual backgrounds in video conferencing apps.

**Key Techniques:**
- Background subtraction
- Color keying (green screen)
- GrabCut segmentation
- Image blending

---

## Methods

### 1. Color Keying (Green Screen)

Best results with a green/blue screen backdrop:

```python
def color_key(frame, background, lower_green, upper_green):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert (foreground is non-green)
    mask = cv2.bitwise_not(mask)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Blur edges for smooth blending
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return blend(frame, background, mask)
```

### 2. Background Subtraction

Learns the background over time:

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True
)

def bg_subtraction(frame, background):
    # Get foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Remove shadows
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    return blend(frame, background, fg_mask)
```

### 3. GrabCut Segmentation

Interactive/semi-automatic segmentation:

```python
def grabcut_segment(frame, background, rect):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create binary mask
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    return blend(frame, background, fg_mask)
```

---

## Blending Function

```python
def blend(foreground, background, mask):
    # Resize background to match
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

    # Convert mask to 3 channels and normalize
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255

    # Blend
    result = foreground.astype(float) * mask_3ch + \
             background.astype(float) * (1 - mask_3ch)

    return result.astype(np.uint8)
```

---

## Method Comparison

| Method | Speed | Quality | Setup Required |
|:-------|:------|:--------|:---------------|
| Color Key | Fast | Excellent | Green screen |
| BG Subtraction | Fast | Good | Static camera |
| GrabCut | Slow | Good | Initial rectangle |

---

## Controls

| Key | Action |
|:----|:-------|
| `1` | Color keying mode |
| `2` | Background subtraction |
| `3` | GrabCut (slow) |
| `n/p` | Next/previous background |
| `+/-` | Adjust color range |
| `r` | Reset background model |
| `s` | Save screenshot |
| `q` | Quit |

---

## Tips for Better Results

1. **Color Keying**: Use evenly lit green/blue screen
2. **BG Subtraction**: Keep camera still, move into frame after start
3. **Lighting**: Avoid shadows on background
4. **Edge smoothing**: Blur mask edges for natural blending

---

## Running the Application

```bash
python curriculum/applications/13_virtual_background.py
```

---

## Official Documentation

- [Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- [GrabCut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html)
