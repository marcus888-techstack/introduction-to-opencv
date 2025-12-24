---
layout: default
title: "04: Features2D"
parent: Modules
nav_order: 4
permalink: /modules/04-features2d
---

# Module 4: Features2D
{: .fs-9 }

Feature detection, description, and matching for image recognition and tracking.
{: .fs-6 .fw-300 }

---

## Topics Covered

- Corner detection (Harris, Shi-Tomasi, FAST)
- Feature descriptors (ORB, SIFT, BRISK, AKAZE)
- Feature matching (BF, FLANN)
- Homography estimation

---

## Algorithm Explanations

### 1. Corner Detection Intuition

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Corner Detection Intuition                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   FLAT REGION           EDGE                   CORNER               │
│                                                                     │
│   ┌───────────┐       ┌───────────┐         ┌───────────┐          │
│   │           │       │███████████│         │███████    │          │
│   │    ───▶   │       │███████████│         │███████    │          │
│   │   ◀─●─▶   │       │──●──▶ ███│         │   ●───────│          │
│   │    ───▶   │       │███████████│         │   │       │          │
│   │           │       │███████████│         │   ▼       │          │
│   └───────────┘       └───────────┘         └───────────┘          │
│                                                                     │
│   No change in        Change in one         Change in ALL          │
│   any direction       direction only        directions             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2. FAST Corner Detection

**FAST Circle of 16 Pixels**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    FAST Detector Circle                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│              16    1    2                                           │
│                ●    ●    ●                                          │
│           15 ●            ● 3                                       │
│                                                                     │
│         14 ●      ┌───┐      ● 4        If N contiguous pixels     │
│                   │ p │                  are ALL brighter OR        │
│         13 ●      └───┘      ● 5        ALL darker than p ± t       │
│                   center                 → CORNER detected          │
│         12 ●                 ● 6                                    │
│                                                                     │
│           11 ●            ● 7                                       │
│                ●    ●    ●                                          │
│              10    9    8                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3. SIFT Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SIFT Algorithm Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   │
│   │  Scale    │──▶│   DoG     │──▶│ Keypoint  │──▶│Orientation│   │
│   │  Space    │   │ Extrema   │   │ Refine    │   │ Assign    │   │
│   └───────────┘   └───────────┘   └───────────┘   └─────┬─────┘   │
│                                                         │          │
│                                   ┌─────────────────────┘          │
│                                   ▼                                 │
│                            ┌───────────┐                           │
│                            │ Descriptor│──▶ 128-dim vector         │
│                            │ Generation│                           │
│                            └───────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**SIFT Descriptor Structure**:
```
16×16 region around keypoint, divided into 4×4 = 16 cells

┌────┬────┬────┬────┐
│ H₁ │ H₂ │ H₃ │ H₄ │    Each cell: 8-bin histogram
├────┼────┼────┼────┤
│ H₅ │ H₆ │ H₇ │ H₈ │    4×4 cells × 8 bins = 128 values
├────┼────┼────┼────┤
│ H₉ │H₁₀│H₁₁│H₁₂│
├────┼────┼────┼────┤               ┌─────────┐
│H₁₃│H₁₄│H₁₅│H₁₆│    ───────────▶ │128-dim  │
└────┴────┴────┴────┘               │ vector  │
                                    └─────────┘
```

---

### 4. Feature Matching

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Feature Matching Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Image A                              Image B                      │
│   ┌─────────────┐                      ┌─────────────┐              │
│   │  ●₁   ●₃    │                      │    ●ₐ  ●ᵦ  │              │
│   │       ●₂    │                      │  ●ᵧ       │              │
│   │  ●₄        │                      │      ●ᵨ    │              │
│   └─────────────┘                      └─────────────┘              │
│        │                                     │                      │
│        ▼                                     ▼                      │
│   ┌─────────────┐                      ┌─────────────┐              │
│   │ Descriptors │                      │ Descriptors │              │
│   └──────┬──────┘                      └──────┬──────┘              │
│          │                                    │                     │
│          └──────────────┬─────────────────────┘                     │
│                         ▼                                           │
│                  ┌─────────────┐                                    │
│                  │   Matcher   │                                    │
│                  │  (BF/FLANN) │                                    │
│                  └──────┬──────┘                                    │
│                         │                                           │
│                         ▼                                           │
│              ┌─────────────────────┐                                │
│              │ Matched pairs:      │                                │
│              │ (1,α), (2,γ), (3,β) │                                │
│              └─────────────────────┘                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5. Lowe's Ratio Test

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Lowe's Ratio Test                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  GOOD MATCH (pass ratio test)      BAD MATCH (fail ratio test)     │
│                                                                     │
│  Best match:    B₃  dist = 15      Best match:    B₇  dist = 45    │
│  2nd best:      B₈  dist = 120     2nd best:      B₂  dist = 52    │
│                                                                     │
│  Ratio: 15/120 = 0.125             Ratio: 45/52 = 0.865            │
│                                                                     │
│  0.125 < 0.75  ✓ KEEP              0.865 > 0.75  ✗ REJECT          │
│                                                                     │
│  Clear winner = distinctive        No clear winner = ambiguous     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 6. RANSAC Homography

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RANSAC Iterations                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Iteration 1:                    Iteration 2:                       │
│  Sample 4 random points          Sample 4 different points          │
│                                                                     │
│  ●─────────────●                 ●─────────────●                    │
│  │  ○   ✗      │                 │  ●   ✗      │                    │
│  │      ○   ✗  │   Count         │  ✗   ●   ●  │   Count            │
│  │  ○     ○    │   inliers       │      ✗   ●  │   inliers          │
│  ●─────────────●   = 6           ●─────────────●   = 3              │
│                                                                     │
│  ● = sampled points              ● = sampled points                 │
│  ○ = inliers (fit H)             ✗ = outliers                       │
│                                                                     │
│  After N iterations:                                                │
│  Keep H with MOST inliers                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Algorithm Comparison

| Algorithm | Speed | Descriptor Size | Best For |
|:----------|:------|:----------------|:---------|
| ORB | Fast | 32 bytes | Real-time, mobile |
| SIFT | Slow | 128 floats | High accuracy |
| BRISK | Fast | 64 bytes | Scale-invariant |
| FAST | Very Fast | N/A (detector only) | Real-time tracking |

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_corners.py` | Harris, Shi-Tomasi, FAST |
| `02_descriptors.py` | ORB, SIFT, BRISK, AKAZE |
| `03_matching.py` | BF matcher, FLANN, ratio test, homography |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
| `cv2.cornerHarris()` | Harris corner detection |
| `cv2.goodFeaturesToTrack()` | Shi-Tomasi corners |
| `cv2.ORB_create()` | ORB detector/descriptor |
| `cv2.SIFT_create()` | SIFT detector/descriptor |
| `cv2.BFMatcher()` | Brute-force matcher |
| `cv2.FlannBasedMatcher()` | FLANN matcher |
| `cv2.findHomography()` | Compute homography |

---

## Further Reading

- [Feature Detection Tutorial](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [SIFT Paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
