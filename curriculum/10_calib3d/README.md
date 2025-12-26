# Module 7: Camera Calibration & 3D Vision

Camera calibration, stereo vision, pose estimation, 3D reconstruction, and Structure from Motion.

## Topics Covered

- Pinhole camera model
- Intrinsic and extrinsic parameters
- Lens distortion & camera calibration
- Stereo vision & depth estimation
- Pose estimation (solvePnP)
- 3D reconstruction & triangulation
- Structure from Motion (SFM)
- Epipolar geometry

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

### 10. Pose Estimation (solvePnP)

**What it does**: Finds camera/object pose from 2D-3D point correspondences.

**PnP Problem Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Perspective-n-Point (PnP)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Known 3D World Points          Camera/Object Pose                  │
│   (e.g., checkerboard)           (Rotation + Translation)           │
│                                                                      │
│      ●───●───●                                                       │
│      │   │   │                        ┌─────┐                       │
│      ●───●───●   ──────────────▶     │  R  │  Rotation (3×3)       │
│      │   │   │      solvePnP         │  t  │  Translation (3×1)    │
│      ●───●───●                        └─────┘                       │
│                                                                      │
│   + 2D Image Points                                                  │
│   + Camera Matrix K                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**solvePnP Algorithms**:
| Flag | Algorithm | Points | Notes |
|------|-----------|--------|-------|
| `SOLVEPNP_ITERATIVE` | Levenberg-Marquardt | 4+ | Default, most robust |
| `SOLVEPNP_P3P` | P3P | Exactly 4 | Returns 4 solutions |
| `SOLVEPNP_EPNP` | EPnP | 4+ | Fast for many points |
| `SOLVEPNP_IPPE` | IPPE | 4+ | For planar objects |

**OpenCV Usage**:
```python
# 3D points in object frame
obj_points = np.array([...], dtype=np.float32)  # Nx3

# 2D points in image
img_points = np.array([...], dtype=np.float32)  # Nx2

# Solve PnP
success, rvec, tvec = cv2.solvePnP(
    obj_points, img_points,
    camera_matrix, dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

# Convert rotation vector to matrix
R, _ = cv2.Rodrigues(rvec)

# With RANSAC for outlier rejection
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    obj_points, img_points,
    camera_matrix, dist_coeffs
)
```

**Rodrigues Representation**:
```
Rotation Vector (rvec):
  - Direction: axis of rotation
  - Magnitude: angle in radians

  rvec = [0.1, 0.2, 0.3]  →  R = cv2.Rodrigues(rvec)[0]
```

---

### 11. Stereo Vision

**What it does**: Estimates depth from two synchronized cameras (stereo pair).

**Stereo Vision Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Stereo Vision Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐   │
│  │  Stereo  │     │  Stereo  │     │ Disparity│     │  Depth   │   │
│  │Calibrate │────▶│ Rectify  │────▶│   Map    │────▶│   Map    │   │
│  └──────────┘     └──────────┘     └──────────┘     └──────────┘   │
│                                                                      │
│  Key Equations:                                                      │
│    disparity = x_left - x_right                                     │
│    depth = (focal_length × baseline) / disparity                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Disparity Algorithms**:
| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| `StereoBM` | Fast | Lower | Real-time |
| `StereoSGBM` | Medium | Higher | Quality |

**OpenCV Usage**:
```python
# Create stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*5,  # Must be divisible by 16
    blockSize=5,
    P1=8 * 3 * 5**2,      # Smoothness penalty
    P2=32 * 3 * 5**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute disparity
disparity = stereo.compute(left_gray, right_gray)

# Convert to depth
depth = (focal_length * baseline) / disparity
```

---

### 12. Triangulation & 3D Reconstruction

**What it does**: Recovers 3D point positions from multiple 2D views.

**Triangulation Geometry**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                       Triangulation                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                           ● P (3D point)                            │
│                          ╱ ╲                                         │
│                         ╱   ╲                                        │
│                  ray 1 ╱     ╲ ray 2                                 │
│                       ╱       ╲                                      │
│            Camera 1 ●─────────● Camera 2                            │
│                     │         │                                      │
│                  ┌──┼──┐   ┌──┼──┐                                  │
│                  │p1●  │   │  ●p2│                                  │
│                  └─────┘   └─────┘                                  │
│                                                                      │
│   Given: p1, p2 (2D), P1, P2 (projection matrices)                  │
│   Find: P (3D point)                                                 │
│                                                                      │
│   x = P @ X  (homogeneous projection)                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**OpenCV Triangulation**:
```python
# Projection matrices (3×4)
P1 = K @ np.hstack([R1, t1])
P2 = K @ np.hstack([R2, t2])

# 2D points (2×N)
pts1 = np.array([[x1, x2, ...], [y1, y2, ...]], dtype=np.float32)
pts2 = np.array([[x1, x2, ...], [y1, y2, ...]], dtype=np.float32)

# Triangulate
points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

# Convert from homogeneous to 3D
points_3d = (points_4d[:3] / points_4d[3]).T
```

**Point Cloud from Stereo**:
```python
# Reproject disparity to 3D
points_3d = cv2.reprojectImageTo3D(disparity, Q)

# Q matrix from stereoRectify
_, _, _, _, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T
)
```

---

### 13. Structure from Motion (SFM)

**What it does**: Reconstructs 3D scene and camera poses from unordered images.

**SFM Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Structure from Motion Pipeline                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Feature Detection & Matching                                     │
│     SIFT/ORB → Match descriptors → Ratio test filter                │
│                    ↓                                                 │
│  2. Fundamental Matrix (uncalibrated)                                │
│     F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)     │
│                    ↓                                                 │
│  3. Essential Matrix (calibrated)                                    │
│     E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC)       │
│     OR: E = K.T @ F @ K                                              │
│                    ↓                                                 │
│  4. Recover Pose                                                     │
│     _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)               │
│                    ↓                                                 │
│  5. Triangulate Points                                               │
│     points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)           │
│                    ↓                                                 │
│  6. Bundle Adjustment (refine all)                                   │
│     Minimize reprojection error over all cameras & points            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Matrices**:
```
Fundamental Matrix F (3×3, rank 2, 7 DoF):
  - Relates pixel coordinates between views
  - p2.T @ F @ p1 = 0
  - No camera calibration needed

Essential Matrix E (3×3, rank 2, 5 DoF):
  - Relates normalized camera coordinates
  - E = K2.T @ F @ K1
  - E = [t]× @ R  (skew-symmetric of translation × rotation)
  - Contains rotation and translation (up to scale)
```

**Epipolar Lines**:
```python
# Compute epipolar lines in image 2 for points in image 1
lines2 = cv2.computeCorrespondEpilines(pts1, 1, F).reshape(-1, 3)

# Draw epipolar lines
for line, pt in zip(lines2, pts1):
    a, b, c = line
    x0, x1 = 0, img.shape[1]
    y0 = int(-c / b)
    y1 = int(-(c + a * x1) / b)
    cv2.line(img2, (x0, y0), (x1, y1), (0, 255, 0), 1)
```

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_camera_calibration.py` | Calibration, undistortion, perspective transform |
| `02_pose_estimation.py` | solvePnP, Rodrigues rotation, AR cube demo |
| `03_stereo_vision.py` | Stereo matching, disparity maps, depth estimation |
| `04_3d_reconstruction.py` | Triangulation, point clouds, PLY export |
| `05_sfm_concepts.py` | Feature matching, F/E matrices, pose recovery |

---

## Key Functions Reference

### Calibration & Undistortion
| Function | Description |
|----------|-------------|
| `cv2.findChessboardCorners()` | Detect checkerboard |
| `cv2.cornerSubPix()` | Refine corner locations |
| `cv2.calibrateCamera()` | Calibrate camera |
| `cv2.undistort()` | Remove distortion |
| `cv2.initUndistortRectifyMap()` | Create undistort maps |

### Stereo Vision
| Function | Description |
|----------|-------------|
| `cv2.stereoCalibrate()` | Calibrate stereo pair |
| `cv2.stereoRectify()` | Compute rectification transforms |
| `cv2.StereoBM_create()` | Block matching stereo |
| `cv2.StereoSGBM_create()` | Semi-global matching |
| `cv2.reprojectImageTo3D()` | Disparity to 3D points |

### Pose Estimation
| Function | Description |
|----------|-------------|
| `cv2.solvePnP()` | Estimate pose from correspondences |
| `cv2.solvePnPRansac()` | PnP with outlier rejection |
| `cv2.Rodrigues()` | Convert rotation vector ↔ matrix |
| `cv2.projectPoints()` | Project 3D to 2D |

### 3D Reconstruction & SFM
| Function | Description |
|----------|-------------|
| `cv2.triangulatePoints()` | Triangulate 3D from 2D pairs |
| `cv2.findFundamentalMat()` | Compute F matrix |
| `cv2.findEssentialMat()` | Compute E matrix |
| `cv2.recoverPose()` | Extract R, t from E |
| `cv2.computeCorrespondEpilines()` | Get epipolar lines |
| `cv2.decomposeEssentialMat()` | Decompose E to R, t |

---

## Calibration Tips

1. **Pattern**: Use asymmetric checkerboard (unequal rows/cols)
2. **Images**: 10-20 images from different angles
3. **Coverage**: Fill the frame, include corners
4. **Angles**: Vary orientation (not just frontal)
5. **Focus**: Ensure pattern is sharp

---

## Further Reading

### Tutorials
- [Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Stereo Depth Map Tutorial](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [Pose Estimation Tutorial](https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html)
- [Epipolar Geometry Tutorial](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html)

### Datasets
- [Middlebury Stereo](https://vision.middlebury.edu/stereo/data/) - Standard stereo benchmarks
- [ETH3D](https://www.eth3d.net/datasets) - Multi-view stereo datasets

### Papers & Books
- [Zhang's Calibration Method](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) - The definitive reference

### Tools
- [COLMAP](https://colmap.github.io/) - State-of-the-art SFM/MVS
- [OpenMVG](https://github.com/openMVG/openMVG) - Open source SFM library
- [Meshroom](https://alicevision.org/) - Photogrammetry GUI
