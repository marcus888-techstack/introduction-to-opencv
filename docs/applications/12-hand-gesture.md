---
layout: default
title: "12: Hand Gesture"
parent: Applications
nav_order: 12
permalink: /applications/12-hand-gesture
---

# Hand Gesture Recognition
{: .fs-9 }

Detect and recognize hand gestures using contour analysis.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/12_hand_gesture.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Recognize hand gestures by counting fingers using skin color segmentation and convexity analysis. No machine learning required!

**Key Techniques:**
- Skin color segmentation (HSV)
- Contour detection
- Convex hull analysis
- Convexity defects

---

## Pipeline

```
Frame → HSV → Skin Mask → Find Contour → Convex Hull → Count Fingers
   ↓      ↓        ↓           ↓             ↓             ↓
[BGR]  [H,S,V]  [Binary]    [Hand]        [Hull]      [Gesture]
```

---

## Skin Color Detection

```python
# HSV ranges for skin detection
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Additional range (red wraps around in HSV)
lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

# Create mask
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
mask = cv2.bitwise_or(mask1, mask2)

# Clean mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=2)
```

---

## Finger Counting with Convexity Defects

```python
# Find largest contour (hand)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hand_contour = max(contours, key=cv2.contourArea)

# Get convex hull
hull = cv2.convexHull(hand_contour)
hull_indices = cv2.convexHull(hand_contour, returnPoints=False)

# Find convexity defects
defects = cv2.convexityDefects(hand_contour, hull_indices)

# Count fingers by analyzing defects
finger_count = 0
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]

    start = tuple(hand_contour[s][0])
    end = tuple(hand_contour[e][0])
    far = tuple(hand_contour[f][0])

    # Calculate angle between fingers
    a = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
    b = np.sqrt((far[0]-start[0])**2 + (far[1]-start[1])**2)
    c = np.sqrt((end[0]-far[0])**2 + (far[1]-far[1])**2)

    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

    # If angle < 90 degrees and defect is deep enough
    if angle <= np.pi/2 and d > 10000:
        finger_count += 1

# Add 1 for thumb
finger_count = min(5, finger_count + 1)
```

---

## Gesture Mapping

| Fingers | Gesture |
|:--------|:--------|
| 0 | Fist |
| 1 | One / Point |
| 2 | Two / Peace |
| 3 | Three |
| 4 | Four |
| 5 | Five / Open Hand |

---

## Visualization

```python
# Draw contour and hull
cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)
cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)

# Draw defect points
for defect in defects:
    s, e, f, d = defect[0]
    if d > 10000:
        far = tuple(hand_contour[f][0])
        cv2.circle(frame, far, 8, (0, 255, 255), -1)
```

---

## Controls

| Key | Action |
|:----|:-------|
| `h/H` | Adjust HSV range |
| `r` | Reset parameters |
| `s` | Save screenshot |
| `q` | Quit |

---

## Tips for Better Detection

1. **Good lighting**: Consistent, even lighting
2. **Plain background**: Avoid skin-colored backgrounds
3. **Hand position**: Keep hand in center, palm facing camera
4. **Calibrate**: Adjust HSV ranges for your skin tone

---

## Running the Application

```bash
python curriculum/applications/12_hand_gesture.py
```

---

## Official Documentation

- [Contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [Convex Hull](https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html)
