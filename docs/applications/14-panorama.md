---
layout: default
title: "14: Panorama Stitcher"
parent: Applications
nav_order: 14
permalink: /applications/14-panorama
---

# Panorama Stitcher
{: .fs-9 }

Create panoramic images by stitching multiple photos.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/14_panorama_stitcher.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Stitch multiple overlapping images into a seamless panorama, like smartphone panorama mode.

**Key Techniques:**
- Feature detection (ORB/SIFT)
- Feature matching
- Homography estimation (RANSAC)
- Image warping and blending

---

## Pipeline

```
Images → Detect Features → Match → Homography → Warp → Blend → Panorama
   ↓           ↓            ↓          ↓          ↓       ↓        ↓
[Multiple]  [Keypoints]  [Pairs]    [Matrix]   [Align]  [Smooth] [Single]
```

---

## Key OpenCV Functions

### Feature Detection

```python
# ORB (fast, free)
detector = cv2.ORB_create(nfeatures=2000)
kp1, des1 = detector.detectAndCompute(gray1, None)
kp2, des2 = detector.detectAndCompute(gray2, None)

# SIFT (slower, more accurate) - requires opencv-contrib
detector = cv2.SIFT_create()
```

### Feature Matching

```python
# For ORB (Hamming distance)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# For SIFT (L2 distance with ratio test)
matcher = cv2.BFMatcher(cv2.NORM_L2)
matches = matcher.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
```

### Homography Estimation

```python
# Extract matched point coordinates
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find homography with RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

### Warping and Blending

```python
# Calculate output canvas size
corners = cv2.perspectiveTransform(corners1, H)
# ... calculate bounds ...

# Warp first image
warped1 = cv2.warpPerspective(img1, H_translated, (output_w, output_h))

# Place second image
warped2 = np.zeros((output_h, output_w, 3), dtype=np.uint8)
warped2[offset_y:offset_y+h2, offset_x:offset_x+w2] = img2

# Blend in overlap region
mask1 = (warped1.sum(axis=2) > 0).astype(float)
mask2 = (warped2.sum(axis=2) > 0).astype(float)
overlap = mask1 * mask2

result = np.where(overlap[:,:,np.newaxis] > 0,
                  (warped1.astype(float) + warped2.astype(float)) / 2,
                  np.maximum(warped1, warped2)).astype(np.uint8)
```

---

## OpenCV Stitcher Class

For easier use, OpenCV provides a built-in Stitcher:

```python
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, panorama = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    cv2.imshow("Panorama", panorama)
else:
    print(f"Stitching failed: {status}")
```

Status codes:
- `cv2.Stitcher_OK` - Success
- `cv2.Stitcher_ERR_NEED_MORE_IMGS` - Not enough images
- `cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL` - Matching failed

---

## Tips for Better Panoramas

1. **Overlap**: 30-50% overlap between images
2. **Rotation**: Rotate camera, don't translate
3. **Exposure**: Keep exposure consistent
4. **Static scenes**: Moving objects cause artifacts
5. **Features**: Include textured areas, avoid blank walls

---

## Controls

| Key | Action |
|:----|:-------|
| `c` | Capture frame (webcam mode) |
| `s` | Stitch captured images |
| `o` | Use OpenCV Stitcher |
| `r` | Reset/clear images |
| `f` | Toggle ORB/SIFT |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/14_panorama_stitcher.py
```

---

## Official Documentation

- [Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [Homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
- [Stitching Module](https://docs.opencv.org/4.x/d1/d46/group__stitching.html)
