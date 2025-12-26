---
layout: default
title: "11: Image Stitching"
parent: Teaching Materials
nav_order: 11
permalink: /teaching-materials/11-stitching
---

# Image Stitching Engine

Guide to panorama creation, homography, and blending techniques.

[Download PDF]({{ site.baseurl }}/teaching_materials/11-image-stitching.pdf){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Topics Covered

- **Stitcher Class** - High-level panorama API
- **Feature Matching** - SIFT vs ORB, BFMatcher vs FLANN
- **Homography** - RANSAC estimation, image warping
- **Blending** - Alpha, feather, multi-band techniques
- **Projections** - Planar, cylindrical, spherical

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_panorama.py` | High-level Stitcher API, basic manual stitch |
| `02_manual_stitching.py` | Step-by-step pipeline: features, matching, RANSAC, warping |
| `03_blending_techniques.py` | Blending comparison: none, alpha, feather, multi-band |
| `04_cylindrical_pano.py` | Cylindrical/spherical projections, wide panoramas |

---

## Key Concepts

### Stitching Pipeline
```
Feature Detection → Feature Matching → Homography → Warping → Blending
```

### Blending Methods

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| No blending | Poor | Fast | Preview only |
| Alpha | Medium | Fast | Similar exposure |
| Feather | Good | Medium | General use |
| Multi-band | Best | Slow | Professional quality |

### Projection Types

| Projection | Use Case |
|------------|----------|
| Planar | Small rotations (<90 deg) |
| Cylindrical | 360 degree horizontal panoramas |
| Spherical | Full 360 x 180 degree VR content |

---

## Key Functions

```python
# High-level API
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, panorama = stitcher.stitch(images)

# Manual pipeline
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
warped = cv2.warpPerspective(img, H, (width, height))

# Blending
blender = cv2.detail.MultiBandBlender()
blender = cv2.detail.FeatherBlender()

# Projections
warper = cv2.PyRotationWarper('cylindrical', focal_length)
```
