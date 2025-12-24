---
layout: default
title: "07: Camera Calibration"
parent: Modules
nav_order: 7
---

# Module 7: Camera Calibration

Camera calibration, distortion correction, and perspective geometry.

## Topics Covered

- Pinhole camera model
- Intrinsic and extrinsic parameters
- Lens distortion
- Camera calibration
- Perspective transform
- Homography

---

## Algorithm Explanations

### 1. Pinhole Camera Model

**What it does**: Models how 3D world points project to 2D image.

**Pinhole Camera Visualization**:
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
│   (In practice, we flip the image plane for convenience)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Projection Equation**:
```
     [u]       [X]
s ×  [v] = K × [R|t] × [Y]
     [1]       [Z]
                [1]
```

Where:
- `(X, Y, Z)`: 3D world point
- `(u, v)`: 2D image point (pixels)
- `s`: Scale factor
- `K`: Camera matrix (intrinsic)
- `[R|t]`: Rotation and translation (extrinsic)

---

### 2. Camera Matrix (Intrinsic Parameters)

**Matrix K**:
```
K = [fₓ  0  cₓ]
    [0  fᵧ  cᵧ]
    [0   0   1]
```

**Parameters**:
| Symbol | Description |
|--------|-------------|
| `fₓ, fᵧ` | Focal length in pixels |
| `cₓ, cᵧ` | Principal point (optical center) |

**Relationship**:
```
fₓ = f × mₓ   (f = focal length in mm, mₓ = pixels per mm)
```

**Pixel Coordinates**:
```
u = fₓ × (X/Z) + cₓ
v = fᵧ × (Y/Z) + cᵧ
```

---

### 3. Extrinsic Parameters

**Rotation Matrix R** (3×3): Camera orientation
**Translation Vector t** (3×1): Camera position

**World to Camera Transform**:
```
[Xc]       [X]
[Yc] = R × [Y] + t
[Zc]       [Z]
```

---

### 4. Lens Distortion

Real lenses introduce distortions. Main types:

**Distortion Types Visualization**:
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
│    Common in             Reference            Common in            │
│    wide-angle                                 telephoto            │
│    lenses                                     lenses               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Radial Distortion
```
x_distorted = x(1 + k₁r² + k₂r⁴ + k₃r⁶)
y_distorted = y(1 + k₁r² + k₂r⁴ + k₃r⁶)

Where: r² = x² + y²
```

**Types**:
- `k > 0`: Barrel distortion (edges bent outward)
- `k < 0`: Pincushion distortion (edges bent inward)

#### Tangential Distortion
```
x_distorted = x + [2p₁xy + p₂(r² + 2x²)]
y_distorted = y + [p₁(r² + 2y²) + 2p₂xy]
```

**Distortion Coefficients** (OpenCV order):
```
dist_coeffs = [k₁, k₂, p₁, p₂, k₃]
```

---

### 5. Calibration Process

**Goal**: Estimate K and distortion coefficients from calibration images.

**Calibration Overview**:
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

**Checkerboard Pattern Detection**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Corner Detection on Pattern                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Image               Detected Corners                       │
│                                                                     │
│   ┌───┬───┬───┬───┐        ┌───┬───┬───┬───┐                       │
│   │▓▓▓│   │▓▓▓│   │        │▓▓▓│   │▓▓▓│   │                       │
│   ├───┼───┼───┼───┤        ├───●───●───●───┤                       │
│   │   │▓▓▓│   │▓▓▓│        │   │▓▓▓│   │▓▓▓│                       │
│   ├───┼───┼───┼───┤  ───▶  ├───●───●───●───┤    ● = detected      │
│   │▓▓▓│   │▓▓▓│   │        │▓▓▓│   │▓▓▓│   │        corner         │
│   ├───┼───┼───┼───┤        ├───●───●───●───┤                       │
│   │   │▓▓▓│   │▓▓▓│        │   │▓▓▓│   │▓▓▓│                       │
│   └───┴───┴───┴───┘        └───┴───┴───┴───┘                       │
│                                                                     │
│   findChessboardCorners()   →  cornerSubPix()  →  calibrateCamera()│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Algorithm**:
```
1. Capture images of calibration pattern (checkerboard)
2. For each image:
   a. Detect pattern corners
   b. Refine corner locations (subpixel)
   c. Store 2D image points
   d. Store corresponding 3D world points

3. Solve for camera parameters:
   - Minimize reprojection error
   - Zhang's method (planar pattern)

4. Compute reprojection error
```

**World Points** (checkerboard):
```python
objp = np.zeros((rows × cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
# Points at Z=0: (0,0,0), (1,0,0), (2,0,0), ...
```

**OpenCV Calibration**:
```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,      # List of 3D points
    imgpoints,      # List of 2D points
    imageSize,      # (width, height)
    None, None      # Initial guess
)
```

**Reprojection Error**:
```
error = (1/N) × Σ ||projected_point - detected_point||²
```
Good calibration: error < 0.5 pixels

---

### 6. Undistortion

**What it does**: Removes lens distortion from images.

**Undistortion Process**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                      Undistortion Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Distorted Image                     Undistorted Image             │
│                                                                     │
│   ╭──────────────╮                    ┌──────────────┐             │
│  ╱   ╭────────╮   ╲      Apply       │   ┌────────┐   │             │
│ │   ╱          ╲   │    Correction   │   │        │   │             │
│ │  │            │  │  ────────────▶  │   │        │   │             │
│ │   ╲          ╱   │   using K &     │   │        │   │             │
│  ╲   ╰────────╯   ╱   dist_coeffs   │   └────────┘   │             │
│   ╰──────────────╯                    └──────────────┘             │
│                                                                     │
│   Barrel distortion                   Straight lines               │
│   (curved edges)                      (corrected)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Remap Method (Faster for Video)**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Remap for Undistortion                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │          One-time computation (slow)                     │      │
│   │  initUndistortRectifyMap(K, dist, size) → mapx, mapy    │      │
│   └─────────────────────────────────────────────────────────┘      │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │          Per-frame application (fast)                    │      │
│   │          remap(image, mapx, mapy) → undistorted         │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
│   mapx[y,x] = source x coordinate    (floating point)              │
│   mapy[y,x] = source y coordinate    (floating point)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Method 1: Direct Undistortion
```python
undistorted = cv2.undistort(image, mtx, dist)
```

#### Method 2: Remap (Faster for Multiple Images)
```python
# Compute maps once
mapx, mapy = cv2.initUndistortRectifyMap(
    mtx, dist, None, new_mtx, size, cv2.CV_32FC1
)

# Apply to any image
undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
```

---

### 7. Perspective Transform

**What it does**: Maps quadrilateral to rectangle (or any 4-point correspondence).

**Perspective Transform Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Perspective Transformation                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Source (skewed view)              Destination (top-down view)    │
│                                                                     │
│         p1●─────────●p2                   ●─────────────●          │
│          ╱           ╲                    │             │          │
│         ╱             ╲      3×3          │             │          │
│        ╱               ╲    Matrix        │             │          │
│       ╱                 ╲   ──────▶       │             │          │
│      ╱                   ╲    H           │             │          │
│     ╱                     ╲               │             │          │
│   p4●───────────────────●p3               ●─────────────●          │
│                                                                     │
│   4 source points   →   getPerspectiveTransform()   →   4 dst pts  │
│                                                                     │
│   Common Use Cases:                                                 │
│   • Document scanning (straighten paper)                           │
│   • Bird's eye view from dashcam                                   │
│   • Sports field analysis                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Point Ordering Convention**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Point Order Matters!                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Standard ordering (clockwise from top-left):                     │
│                                                                     │
│       [0]●─────────────────●[1]        src = [[x0,y0],             │
│          │                 │                  [x1,y1],             │
│          │                 │                  [x2,y2],             │
│          │                 │                  [x3,y3]]             │
│          │                 │                                        │
│       [3]●─────────────────●[2]        dst = [[0,0],               │
│                                               [w,0],               │
│                                               [w,h],               │
│                                               [0,h]]               │
│                                                                     │
│   Wrong order → distorted/flipped output!                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Homography Matrix H** (3×3):
```
[x']   [h₁₁ h₁₂ h₁₃] [x]
[y'] = [h₂₁ h₂₂ h₂₃] [y]
[w']   [h₃₁ h₃₂ h₃₃] [1]

x' = (h₁₁x + h₁₂y + h₁₃) / (h₃₁x + h₃₂y + h₃₃)
y' = (h₂₁x + h₂₂y + h₂₃) / (h₃₁x + h₃₂y + h₃₃)
```

**From 4 Point Correspondences**:
```python
# Source points (corners of skewed rectangle)
src = np.float32([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])

# Destination points (corners of output rectangle)
dst = np.float32([[0,0], [w,0], [w,h], [0,h]])

# Get transform matrix
M = cv2.getPerspectiveTransform(src, dst)

# Apply transform
warped = cv2.warpPerspective(image, M, (w, h))
```

---

### 8. Homography

**What it does**: Relates two views of a planar surface.

**Homography Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Homography Between Views                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Camera 1 View                       Camera 2 View                │
│                                                                     │
│   ┌───────────────┐                   ┌───────────────┐            │
│   │ ┌───┐         │                   │         ┌───┐ │            │
│   │ │ A │  planar │      H            │ planar  │ A │ │            │
│   │ └───┘  surface│  ◀───────▶        │ surface └───┘ │            │
│   │    ┌───┐      │   3×3 matrix      │      ┌───┐    │            │
│   │    │ B │      │                   │      │ B │    │            │
│   │    └───┘      │                   │      └───┘    │            │
│   └───────────────┘                   └───────────────┘            │
│                                                                     │
│   p₂ = H × p₁     (homogeneous coordinates)                        │
│                                                                     │
│   Key insight: Homography only works for planar surfaces or        │
│                pure camera rotation (no translation)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Applications**:
- Image stitching
- Augmented reality
- Perspective correction
- Object pose estimation

**Degrees of Freedom**: 8 (9 elements - 1 scale)

**Minimum Points**: 4 correspondences

**RANSAC Estimation**:
```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# mask indicates inliers
```

**Decomposition** (to extract R and t):
```python
num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)
```

---

### 9. Fundamental and Essential Matrices

**Epipolar Geometry**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                       Epipolar Geometry                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                           P (3D point)                              │
│                              ●                                      │
│                             ╱╲                                      │
│                            ╱  ╲                                     │
│                           ╱    ╲                                    │
│                          ╱      ╲                                   │
│                         ╱        ╲                                  │
│      Camera 1         ╱          ╲         Camera 2                │
│         ●────────────╱────────────╲────────────●                   │
│        ╱│           ╱              ╲           │╲                  │
│       ╱ │          ╱                ╲          │ ╲                 │
│      ╱  │       p1●                  ●p2       │  ╲                │
│     ╱   │      ╱                        ╲      │   ╲               │
│    ╱    │  Image 1                  Image 2   │    ╲              │
│   ╱     └──┬─────────────────────────────┬────┘     ╲             │
│  e1        │     epipolar line l2        │          e2             │
│ (epipole)  │◀───────────────────────────▶│      (epipole)          │
│            │   p2 lies on this line!     │                         │
│                                                                     │
│   Given p1, the corresponding p2 MUST lie on epipolar line l2     │
│   This constrains the search to 1D instead of 2D!                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Essential Matrix E
**Relates**: Normalized camera coordinates
```
x₂ᵀ E x₁ = 0
E = [t]ₓ R

Where [t]ₓ is the skew-symmetric matrix of t
```

#### Fundamental Matrix F
**Relates**: Pixel coordinates
```
p₂ᵀ F p₁ = 0
F = K₂⁻ᵀ E K₁⁻¹
```

**Epipolar Constraint**: Corresponding points lie on epipolar lines.

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_camera_calibration.py` | Calibration, undistortion, perspective transform |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.findChessboardCorners()` | Detect checkerboard |
| `cv2.cornerSubPix()` | Refine corner locations |
| `cv2.drawChessboardCorners()` | Visualize corners |
| `cv2.calibrateCamera()` | Calibrate camera |
| `cv2.undistort()` | Remove distortion |
| `cv2.initUndistortRectifyMap()` | Create undistort maps |
| `cv2.remap()` | Apply mapping |
| `cv2.getPerspectiveTransform()` | Get 3×3 transform |
| `cv2.warpPerspective()` | Apply perspective warp |
| `cv2.findHomography()` | Estimate homography |
| `cv2.projectPoints()` | Project 3D to 2D |

---

## Calibration Tips

1. **Pattern**: Use asymmetric checkerboard (unequal rows/cols)
2. **Images**: 10-20 images from different angles
3. **Coverage**: Fill the frame, include corners
4. **Angles**: Vary orientation (not just frontal)
5. **Focus**: Ensure pattern is sharp

---

## Further Reading

- [Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Zhang's Method Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
- [Multiple View Geometry Book](https://www.robots.ox.ac.uk/~vgg/hzbook/)
