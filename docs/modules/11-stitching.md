---
layout: default
title: "11: Image Stitching"
parent: Modules
nav_order: 11
permalink: /modules/11-stitching
---

# Module 11: Image Stitching
{: .fs-9 }

Creating panoramic images from multiple overlapping photographs.
{: .fs-6 .fw-300 }

---

## Topics Covered

- High-level Stitcher API
- Manual stitching pipeline
- Feature-based alignment
- Image warping
- Blending techniques

---

## Algorithm Explanations

### 1. Panorama Stitching Overview

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
|:-----|:--------|
| `Stitcher_OK` | Success |
| `Stitcher_ERR_NEED_MORE_IMGS` | Not enough images |
| `Stitcher_ERR_HOMOGRAPHY_EST_FAIL` | Homography failed |
| `Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL` | Camera calibration failed |

---

### 3. Homography Estimation

**Homography Matrix H** (3×3):
```
Transforms points from image 1 to image 2:

[x']   [h₁₁ h₁₂ h₁₃] [x]
[y'] = [h₂₁ h₂₂ h₂₃] [y]
[w']   [h₃₁ h₃₂ h₃₃] [1]
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

---

### 4. Image Warping

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

**Warping Types**:

| Warper | Description | Use Case |
|:-------|:------------|:---------|
| Plane | Planar projection | Small rotations |
| Cylindrical | Cylindrical surface | 360° horizontal |
| Spherical | Spherical surface | Full 360° × 180° |
| Fisheye | Fisheye correction | Wide-angle lenses |

---

### 5. Blending Techniques

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

---

### 6. Multi-Band Blending (Laplacian Pyramid)

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

---

## Comparison

| Blending Method | Quality | Speed | Artifacts |
|:----------------|:--------|:------|:----------|
| No blending | Poor | Fast | Visible seams |
| Alpha | Medium | Fast | Ghosting, gradient |
| Feather | Good | Medium | Slight blur |
| Multi-band | Best | Slow | Minimal |

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_panorama.py` | Stitcher API, manual pipeline, blending |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
| `cv2.Stitcher_create()` | Create stitcher object |
| `stitcher.stitch()` | Stitch images |
| `cv2.findHomography()` | Compute homography matrix |
| `cv2.warpPerspective()` | Apply perspective transform |
| `cv2.perspectiveTransform()` | Transform points |
| `cv2.detail.MultiBandBlender()` | Multi-band blender |
| `cv2.detail.FeatherBlender()` | Feather blender |

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
