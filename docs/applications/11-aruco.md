---
layout: default
title: "11: ArUco Markers"
parent: Applications
nav_order: 11
permalink: /applications/11-aruco
---

# ArUco Marker Detection
{: .fs-9 }

Detect and track ArUco markers for augmented reality.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/11_aruco_detection.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

ArUco markers are binary square fiducial markers used in computer vision for camera pose estimation and augmented reality applications.

**Key Techniques:**
- ArUco dictionary generation
- Marker detection
- Pose estimation
- 3D axis drawing

---

## ArUco Dictionaries

| Dictionary | Markers | Best For |
|:-----------|:--------|:---------|
| DICT_4X4_50 | 50 | Small markers, close range |
| DICT_5X5_100 | 100 | Medium markers |
| DICT_6X6_250 | 250 | Large markers, far range |
| DICT_ARUCO_ORIGINAL | 1024 | Original ArUco library |

---

## Key OpenCV Functions

```python
# Get dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect markers
corners, ids, rejected = detector.detectMarkers(gray)

# Draw detected markers
cv2.aruco.drawDetectedMarkers(frame, corners, ids)

# Generate a marker image
marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id=0, sidePixels=200)
```

---

## Pose Estimation

```python
# Camera matrix (should be from calibration)
focal_length = frame.shape[1]
center = (frame.shape[1] / 2, frame.shape[0] / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# Marker size in meters
marker_size = 0.05

# Define 3D points
obj_points = np.array([
    [-marker_size/2, marker_size/2, 0],
    [marker_size/2, marker_size/2, 0],
    [marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# Solve PnP for each marker
for corner in corners:
    success, rvec, tvec = cv2.solvePnP(
        obj_points, corner[0], camera_matrix, dist_coeffs
    )
    if success:
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
```

---

## Generate Marker Board

```python
def generate_board(rows=2, cols=3, marker_size=100, margin=20):
    board_width = cols * marker_size + (cols + 1) * margin
    board_height = rows * marker_size + (rows + 1) * margin
    board = np.ones((board_height, board_width), dtype=np.uint8) * 255

    marker_id = 0
    for r in range(rows):
        for c in range(cols):
            marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
            x = margin + c * (marker_size + margin)
            y = margin + r * (marker_size + margin)
            board[y:y+marker_size, x:x+marker_size] = marker
            marker_id += 1

    return board
```

---

## Applications

- **Augmented Reality**: Overlay 3D objects on markers
- **Robot Navigation**: Localization using markers
- **Camera Calibration**: Known-size markers
- **Object Tracking**: Track marker position and orientation

---

## Controls

| Key | Action |
|:----|:-------|
| `1-4` | Change dictionary |
| `p` | Toggle pose estimation |
| `g` | Generate and save marker |
| `s` | Save screenshot |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/11_aruco_detection.py
```

**Note:** Requires `opencv-contrib-python` for ArUco module.

---

## Official Documentation

- [ArUco Detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [ArUco Module](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html)
