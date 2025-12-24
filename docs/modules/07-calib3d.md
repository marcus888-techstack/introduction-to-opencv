---
layout: default
title: "07: Camera Calibration"
parent: Modules
nav_order: 7
permalink: /modules/07-calib3d
---

# Module 7: Camera Calibration
{: .fs-9 }

Camera calibration, distortion correction, and perspective geometry.
{: .fs-6 .fw-300 }

---

## Topics Covered

- Pinhole camera model
- Intrinsic and extrinsic parameters
- Lens distortion
- Camera calibration
- Perspective transform

---

## Algorithm Explanations

### 1. Pinhole Camera Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Pinhole Camera Model                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   World                  Pinhole              Image Plane           │
│                          (focal point)                              │
│      ●────────────────────●────────────────────┬─────┐             │
│   (X,Y,Z)                 │                    │  ●  │ (u,v)       │
│      3D point             │                    │     │ 2D point    │
│                           │                    │     │             │
│                           │        f           │     │             │
│                           │◀──────────────────▶│     │             │
│                           │    focal length    │     │             │
│                                                └─────┘              │
│                                                                     │
│   Light rays pass through pinhole → inverted image on plane        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Camera Matrix (Intrinsic Parameters)**:
```
K = [fₓ  0  cₓ]
    [0  fᵧ  cᵧ]
    [0   0   1]
```

| Symbol | Description |
|:-------|:------------|
| `fₓ, fᵧ` | Focal length in pixels |
| `cₓ, cᵧ` | Principal point (optical center) |

---

### 2. Lens Distortion

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Lens Distortion Types                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    Barrel (k > 0)        No Distortion       Pincushion (k < 0)    │
│                                                                     │
│    ╭──────────╮          ┌──────────┐          ╱──────────╲        │
│   ╱            ╲         │          │         ╱            ╲       │
│  │              │        │          │        ╲              ╱      │
│  │              │        │          │        ╲              ╱      │
│   ╲            ╱         │          │         ╱            ╲       │
│    ╰──────────╯          └──────────┘          ╲──────────╱        │
│                                                                     │
│    Edges bow             Perfect grid         Edges pinch          │
│    outward               (ideal)              inward               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Radial Distortion**:
```
x_distorted = x(1 + k₁r² + k₂r⁴ + k₃r⁶)
y_distorted = y(1 + k₁r² + k₂r⁴ + k₃r⁶)
```

**Distortion Coefficients** (OpenCV order):
```
dist_coeffs = [k₁, k₂, p₁, p₂, k₃]
```

---

### 3. Calibration Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Camera Calibration Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Capture    │    │  Detect     │    │  Refine     │             │
│  │  Multiple   │───▶│  Corners    │───▶│  Subpixel   │             │
│  │  Images     │    │             │    │  Accuracy   │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                                      │                     │
│        │                                      ▼                     │
│        │              ┌─────────────────────────────────┐          │
│        │              │     Known 3D World Points       │          │
│        │              │  (0,0,0) (1,0,0) (2,0,0) ...    │          │
│        │              │  Checkerboard at Z = 0          │          │
│        │              └─────────────────────────────────┘          │
│        │                              │                             │
│        ▼                              ▼                             │
│  ┌─────────────┐    ┌─────────────────────────────────┐            │
│  │  Intrinsic  │◀───│   Zhang's Calibration Method    │            │
│  │  Matrix K   │    │   (Minimize Reprojection Error) │            │
│  │  Distortion │    └─────────────────────────────────┘            │
│  │  Coeffs     │                                                    │
│  └─────────────┘                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 4. Perspective Transform

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Perspective Transform                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Image               Transform Matrix        Output Image   │
│   (Skewed)                                          (Rectified)    │
│                                                                     │
│   ●─────────●                  [h₁₁ h₁₂ h₁₃]       ●───────────●   │
│   │         │                  [h₂₁ h₂₂ h₂₃]       │           │   │
│   │    ◊    │       ───▶       [h₃₁ h₃₂ h₃₃]   ───▶│     ◊     │   │
│   │         │                                       │           │   │
│   ●─────────●                                       ●───────────●   │
│                                                                     │
│   4 source points             Homography           4 dest points   │
│                               matrix                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_camera_calibration.py` | Calibration, undistortion, perspective transform |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
| `cv2.calibrateCamera()` | Camera calibration |
| `cv2.undistort()` | Remove lens distortion |
| `cv2.findChessboardCorners()` | Detect calibration pattern |
| `cv2.getPerspectiveTransform()` | Get transform matrix |
| `cv2.warpPerspective()` | Apply perspective transform |

---

## Further Reading

- [Camera Calibration](https://docs.opencv.org/4.x/d9/db7/tutorial_py_table_of_contents_calib3d.html)
