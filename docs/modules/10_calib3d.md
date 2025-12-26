---
layout: default
title: "10: 3D Vision"
parent: Modules
nav_order: 10
permalink: /modules/10-calib3d
---

# Module 10: Camera Calibration & 3D Vision

Camera calibration, stereo vision, 3D reconstruction, and Structure from Motion.

## Topics Covered

- Camera calibration and undistortion
- Pose estimation with PnP
- Stereo vision and disparity maps
- 3D reconstruction and point clouds
- Structure from Motion (SfM)

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_camera_calibration.py` | Calibration, undistortion, perspective transform |
| `02_pose_estimation.py` | 3D pose estimation with PnP |
| `03_stereo_vision.py` | Stereo calibration, rectification, disparity maps |
| `04_3d_reconstruction.py` | Depth estimation and point cloud generation |
| `05_sfm_concepts.py` | Structure from Motion fundamentals |

---

## Key Concepts

### Camera Matrix K
```
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
```

### Stereo Pipeline
```
Calibration → Rectification → Matching → Disparity → Depth
```

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.findChessboardCorners()` | Detect calibration pattern |
| `cv2.calibrateCamera()` | Calibrate camera |
| `cv2.undistort()` | Remove lens distortion |
| `cv2.solvePnP()` | Estimate 3D pose |
| `cv2.stereoCalibrate()` | Calibrate stereo pair |
| `cv2.stereoRectify()` | Rectify stereo images |
| `cv2.StereoBM_create()` | Block matching disparity |
| `cv2.StereoSGBM_create()` | Semi-global matching |

---

## Further Reading

- [Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Stereo Vision Tutorial](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [Multiple View Geometry Book](https://www.robots.ox.ac.uk/~vgg/hzbook/)
