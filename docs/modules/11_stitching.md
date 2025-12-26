---
layout: default
title: "11: Stitching"
parent: Modules
nav_order: 11
permalink: /modules/11-stitching
---

# Module 11: Image Stitching

Creating panoramic images from multiple overlapping photographs.

## Topics Covered

- High-level Stitcher API
- Manual stitching pipeline
- Feature-based alignment
- Image warping
- Blending techniques

---

## Algorithm Explanations

### 1. Panorama Stitching Overview

**Panorama Stitching Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Panorama Stitching                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Images (with overlap)                                      │
│                                                                     │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐                       │
│   │   Image   │ │   Image   │ │   Image   │                       │
│   │     1     │◀─▶│     2     │◀─▶│     3     │                       │
│   │           │ │           │ │           │                       │
│   └───────────┘ └───────────┘ └───────────┘                       │
│        ↑overlap↑     ↑overlap↑                                     │
│        (20-40%)      (20-40%)                                      │
│                                                                     │
│                         │                                          │
│                         ▼                                          │
│                                                                     │
│   Output: Seamless Panorama                                        │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │                                                              │ │
│   │                    Blended Panorama                         │ │
│   │                                                              │ │
│   └─────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Pipeline**:
```
1. Feature Detection → Find distinctive points in each image
2. Feature Matching  → Find correspondences between images
3. Homography        → Compute geometric transformation
4. Warping          → Transform images to common plane
5. Blending         → Seamlessly combine warped images
```

---

### 2. High-Level Stitcher API

**OpenCV Stitcher**:
```python
# Create stitcher
stitcher = cv2.Stitcher_create(mode)

# Modes:
cv2.Stitcher_PANORAMA  # For camera rotation (default)
cv2.Stitcher_SCANS     # For flat document scans

# Stitch images
status, panorama = stitcher.stitch(images)
```

**Status Codes**:
| Code | Meaning |
|------|---------|
| `Stitcher_OK` | Success |
| `Stitcher_ERR_NEED_MORE_IMGS` | Not enough images |
| `Stitcher_ERR_HOMOGRAPHY_EST_FAIL` | Homography failed |
| `Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL` | Camera calibration failed |

---

### 3. Feature Detection and Matching

**Step 1: Detect Features**:
```python
# Use SIFT or ORB
detector = cv2.SIFT_create()
# or
detector = cv2.ORB_create(nfeatures=1000)

keypoints, descriptors = detector.detectAndCompute(image, None)
```

**Step 2: Match Features**:
```python
# Brute-force or FLANN matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

**Minimum Matches**: Need at least 4 point correspondences for homography.

---

### 4. Homography Estimation

**Homography Matrix H** (3×3):
```
Transforms points from image 1 to image 2:

[x']   [h₁₁ h₁₂ h₁₃] [x]
[y'] = [h₂₁ h₂₂ h₂₃] [y]
[w']   [h₃₁ h₃₂ h₃₃] [1]

Normalized:
x' = (h₁₁x + h₁₂y + h₁₃) / (h₃₁x + h₃₂y + h₃₃)
y' = (h₂₁x + h₂₂y + h₂₃) / (h₃₁x + h₃₂y + h₃₃)
```

**Degrees of Freedom**: 8 (9 elements - 1 for scale)

**RANSAC Estimation**:
```python
# Extract matched point coordinates
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find homography with RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# mask indicates inliers (1) and outliers (0)
```

**RANSAC Algorithm**:
```
1. Randomly select 4 point correspondences
2. Compute homography from these 4 points
3. Count inliers (points that fit H within threshold)
4. Repeat for N iterations
5. Keep H with most inliers
6. Recompute H using all inliers
```

**Number of Iterations**:
```
N = log(1 - p) / log(1 - wⁿ)

Where:
  p = desired success probability (e.g., 0.99)
  w = inlier ratio
  n = points per sample (4 for homography)
```

---

### 5. Image Warping

**Warping to Common Plane**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Image Warping for Stitching                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Original Images            Warped to Common Plane                │
│                                                                     │
│   ┌───────┐ ┌───────┐        ╱─────────────────────────────╲      │
│   │       │ │       │       ╱     ┌───────────────────┐     ╲     │
│   │ Img 1 │ │ Img 2 │  ──▶ │     │      Warped 1      │      │    │
│   │       │ │       │       │   ╱─┴─────────┬─────────┴──╲   │    │
│   └───────┘ └───────┘       │  │   Overlap  │  Warped 2   │  │    │
│                             │   ╲──────────┴────────────╱   │    │
│                              ╲                              ╱     │
│                               ╲────────────────────────────╱      │
│                                                                     │
│   Homography H transforms Image 1's coordinates to align          │
│   with Image 2's coordinate system                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Perspective Transform**:
```python
# Warp image using homography
warped = cv2.warpPerspective(image, H, (width, height))
```

**Canvas Size Calculation**:
```python
# Transform image corners to find output dimensions
h, w = img.shape[:2]
corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
transformed = cv2.perspectiveTransform(corners, H)

# Find bounding box
min_x, min_y = transformed.min(axis=0).flatten()
max_x, max_y = transformed.max(axis=0).flatten()

# Create translation matrix if needed
translation = np.array([
    [1, 0, -min_x],
    [0, 1, -min_y],
    [0, 0, 1]
])
```

**Warping Types**:
| Warper | Description | Use Case |
|--------|-------------|----------|
| Plane | Planar projection | Small rotations |
| Cylindrical | Cylindrical surface | 360° horizontal |
| Spherical | Spherical surface | Full 360° × 180° |
| Fisheye | Fisheye correction | Wide-angle lenses |

---

### 6. Blending Techniques

**Blending Comparison**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Blending Methods Comparison                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   No Blending           Alpha Blending        Multi-Band           │
│                                                                     │
│   ┌───────┬───────┐     ┌───────┬───────┐     ┌─────────────────┐ │
│   │▓▓▓▓▓▓▓│░░░░░░░│     │▓▓▓▓▒▒▒░░░░░░░│     │▓▓▓▓▓▒▒░░░░░░░░░│ │
│   │▓▓▓▓▓▓▓│░░░░░░░│     │▓▓▓▓▒▒▒░░░░░░░│     │▓▓▓▓▓▒▒░░░░░░░░░│ │
│   │▓▓▓▓▓▓▓│░░░░░░░│     │▓▓▓▓▒▒▒░░░░░░░│     │▓▓▓▓▓▒▒░░░░░░░░░│ │
│   └───────┴───────┘     └───────────────┘     └─────────────────┘ │
│        │                     ▒▒▒                    ▒▒             │
│   Visible seam         Gradient blend        Seamless blend       │
│   (hard edge)          (may ghost)           (preserves edges)    │
│                                                                     │
│   Speed: Fast          Speed: Fast           Speed: Slow          │
│   Quality: Poor        Quality: Medium       Quality: Best        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Multi-Band Blending (Laplacian Pyramid)**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Band Blending                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Image 1          Mask           Image 2                          │
│   ┌─────┐        ┌─────┐        ┌─────┐                           │
│   │▓▓▓▓▓│        │█████│        │░░░░░│                           │
│   │▓▓▓▓▓│        │██░░░│        │░░░░░│                           │
│   └─────┘        └─────┘        └─────┘                           │
│      │              │              │                               │
│      ▼              ▼              ▼                               │
│   ┌─────┐        ┌─────┐        ┌─────┐    Level 0 (full res)     │
│   │ L1_0│        │ M_0 │        │ L2_0│    High frequency         │
│   └─────┘        └─────┘        └─────┘                           │
│   ┌───┐          ┌───┐          ┌───┐      Level 1                │
│   │L1_1│         │M_1│          │L2_1│     Mid frequency          │
│   └───┘          └───┘          └───┘                             │
│   ┌─┐            ┌─┐            ┌─┐        Level 2                │
│   │ │            │ │            │ │        Low frequency          │
│   └─┘            └─┘            └─┘                               │
│                                                                     │
│   Blend at each level: B_k = M_k × L1_k + (1-M_k) × L2_k          │
│   Collapse pyramid to get final result                             │
│                                                                     │
│   Key insight: Blend low frequencies broadly, high freq narrowly   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### No Blending (Simple Copy)
```
Just overlay images - visible seams
```

#### Alpha Blending
**Linear interpolation** in overlap region:
```
Result(x) = α × I₁(x) + (1 - α) × I₂(x)

Where α varies linearly across overlap:
α = distance_from_right_edge / overlap_width
```

```python
for i in range(overlap_width):
    alpha = i / overlap_width
    result[:, x+i] = (1 - alpha) * img1[:, x+i] + alpha * img2[:, i]
```

#### Feather Blending
**Distance-weighted** combination:
```
Weight(p) = min(dist_to_edge₁, dist_to_edge₂)

Result(p) = Σᵢ wᵢ(p) × Iᵢ(p) / Σᵢ wᵢ(p)
```

OpenCV:
```python
blender = cv2.detail.FeatherBlender()
blender.setSharpness(0.02)
```

#### Multi-Band Blending
**Best quality** - blends different frequencies separately.

**Algorithm**:
```
1. Build Laplacian pyramid for each image:
   L_k = G_k - expand(G_{k+1})

2. Build Gaussian pyramid for mask:
   M_k = reduce(M_{k-1})

3. Blend at each level:
   B_k = M_k × L₁_k + (1 - M_k) × L₂_k

4. Reconstruct from blended pyramid:
   Result = collapse(B)
```

**Laplacian Pyramid**:
```
Level k stores high-frequency details:
L_k = G_k - upsample(G_{k+1})
```

OpenCV:
```python
blender = cv2.detail.MultiBandBlender()
blender.setNumBands(5)  # Number of pyramid levels
```

---

### 7. Seam Finding

**What it does**: Finds optimal cut line through overlap to minimize visibility.

**Graph Cut Approach**:
```
1. Model overlap as graph
2. Edge weights = color difference
3. Find minimum cut (min-cut/max-flow)
4. Seam follows minimum cut
```

**OpenCV Seam Finders**:
```python
seam_finder = cv2.detail.GraphCutSeamFinder('COST_COLOR')
# or
seam_finder = cv2.detail.DpSeamFinder('COLOR')  # Dynamic programming
seam_finder = cv2.detail.VoronoiSeamFinder()    # Voronoi diagram
```

---

### 8. Manual Stitching Pipeline

**Complete Example**:
```python
def stitch_two_images(img1, img2):
    # 1. Feature detection
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # 2. Feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # 3. Ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # 4. Extract points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 5. Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 6. Calculate output size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H)

    all_corners = np.concatenate([
        corners1_transformed,
        np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    ])

    min_x, min_y = all_corners.min(axis=0).flatten()
    max_x, max_y = all_corners.max(axis=0).flatten()

    # 7. Translation matrix
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # 8. Warp and combine
    output_size = (int(max_x - min_x), int(max_y - min_y))
    warped1 = cv2.warpPerspective(img1, translation @ H, output_size)

    # Place second image
    x_off, y_off = int(-min_x), int(-min_y)
    warped1[y_off:y_off+h2, x_off:x_off+w2] = img2

    return warped1
```

---

### 9. Stitcher Configuration

**Configure Components**:
```python
stitcher = cv2.Stitcher_create()

# Feature detector
stitcher.setFeaturesFinder(cv2.detail.OrbFeaturesFinder())

# Warper
stitcher.setWarper(cv2.PyRotationWarper('spherical', 1.0))

# Blender
stitcher.setBlender(cv2.detail.MultiBandBlender())

# Resolution settings
stitcher.setRegistrationResol(0.6)  # Matching resolution
stitcher.setCompositingResol(-1)    # Output resolution (-1 = original)
stitcher.setSeamEstimationResol(0.1)
```

---

## Comparison

| Blending Method | Quality | Speed | Artifacts |
|-----------------|---------|-------|-----------|
| No blending | Poor | Fast | Visible seams |
| Alpha | Medium | Fast | Ghosting, gradient |
| Feather | Good | Medium | Slight blur |
| Multi-band | Best | Slow | Minimal |

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_panorama.py` | High-level Stitcher API, basic manual stitch |
| `02_manual_stitching.py` | Step-by-step pipeline: features, matching, RANSAC, warping |
| `03_blending_techniques.py` | Blending comparison: none, alpha, feather, multi-band |
| `04_cylindrical_pano.py` | Cylindrical/spherical projections, wide panoramas |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.Stitcher_create()` | Create stitcher object |
| `stitcher.stitch()` | Stitch images |
| `cv2.SIFT_create()` | SIFT feature detector |
| `cv2.ORB_create()` | ORB feature detector (faster) |
| `cv2.BFMatcher()` | Brute-force feature matcher |
| `cv2.FlannBasedMatcher()` | Fast approximate matcher |
| `cv2.drawMatches()` | Visualize feature matches |
| `cv2.findHomography()` | Compute homography with RANSAC |
| `cv2.warpPerspective()` | Apply perspective transform |
| `cv2.perspectiveTransform()` | Transform points |
| `cv2.pyrDown()` / `cv2.pyrUp()` | Build image pyramids |
| `cv2.detail.MultiBandBlender()` | Multi-band (best quality) |
| `cv2.detail.FeatherBlender()` | Distance-weighted blending |
| `cv2.PyRotationWarper()` | Cylindrical/spherical projection |

---

## Tips for Good Stitching

1. **Overlap**: Ensure 20-40% overlap between images
2. **Rotation**: Rotate camera around optical center
3. **Exposure**: Keep consistent exposure
4. **Movement**: Avoid parallax (moving objects)
5. **Features**: Ensure textured, feature-rich regions

---

## Further Reading

- [Stitching Module](https://docs.opencv.org/4.x/d1/d46/group__stitching.html)
- [Panorama Tutorial](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html)
- [Multi-band Blending Paper](http://persci.mit.edu/pub_pdfs/spline83.pdf)
