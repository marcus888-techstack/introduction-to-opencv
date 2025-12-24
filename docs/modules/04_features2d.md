---
layout: default
title: "04: Features2D"
parent: Modules
nav_order: 4
---

# Module 4: Features2D

Feature detection, description, and matching for image recognition and tracking.

## Topics Covered

- Corner detection (Harris, Shi-Tomasi, FAST)
- Feature descriptors (ORB, SIFT, BRISK, AKAZE)
- Feature matching (BF, FLANN)
- Homography estimation

---

## Algorithm Explanations

### 1. Harris Corner Detection

**What it does**: Detects corners by analyzing intensity changes in all directions.

**Intuition - Why Corners are Special**:
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
│   λ₁ ≈ 0, λ₂ ≈ 0      λ₁ >> λ₂             λ₁ ≈ λ₂ (both large)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Mathematical Foundation**:

For a shift `(u, v)`, the intensity change is:
```
E(u,v) = Σ w(x,y) × [I(x+u, y+v) - I(x,y)]²
```

Using Taylor expansion:
```
E(u,v) ≈ [u v] M [u]
                [v]
```

**Structure Matrix M**:
```
M = Σ w(x,y) [Iₓ²    IₓIᵧ]
             [IₓIᵧ   Iᵧ² ]
```

Where `Iₓ`, `Iᵧ` are image gradients.

**Eigenvalue Classification**:
```
                    λ₂
                     ▲
                     │
                     │  ┌─────────────┐
                     │  │   CORNER    │  λ₁ ≈ λ₂ (both large)
                     │  │   R >> 0    │
                     │  └─────────────┘
                     │         ╱
          EDGE       │        ╱
          R < 0      │       ╱
       ┌─────────┐   │      ╱
       │         │   │     ╱
       │         │   │    ╱
       └─────────┘   │   ╱
                     │  ╱  FLAT
                     │ ╱   R ≈ 0
              ───────┼─────────────────▶ λ₁
                     │
```

**Corner Response Function**:
```
R = det(M) - k × trace(M)²
  = λ₁λ₂ - k(λ₁ + λ₂)²
```

Where:
- `λ₁, λ₂`: eigenvalues of M
- `k`: Harris parameter (typically 0.04-0.06)

**Classification**:
| Condition | Type |
|-----------|------|
| Both λ small | Flat region |
| One λ large | Edge |
| Both λ large | Corner |

**OpenCV**: `cv2.cornerHarris(gray, blockSize, ksize, k)`

---

### 2. Shi-Tomasi (Good Features to Track)

**Improvement over Harris**:
```
R = min(λ₁, λ₂)
```

Corner if `min(λ₁, λ₂) > threshold`

**Advantages**:
- More stable corner selection
- Direct quality measure
- Built-in non-maximum suppression

**OpenCV**:
```python
corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100,
    qualityLevel=0.01,   # Min quality relative to best
    minDistance=10       # Min pixels between corners
)
```

---

### 3. FAST (Features from Accelerated Segment Test)

**What it does**: Extremely fast corner detection using circle test.

**Algorithm**:
1. Consider circle of 16 pixels around candidate point p
2. If N contiguous pixels are all brighter or all darker than p ± threshold:
   - Point is a corner

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

**Corner Detection Example**:
```
Threshold t = 20, p = 100

    Brighter pixels (> 120):     All contiguous?
         ●    ●    ●              YES! This is
        ○            ●            a corner
       ○              ●
       ○      p       ●           9 contiguous pixels
       ○              ●           are all brighter
        ○            ●
         ○    ○    ●

    ● = brighter (I > p + t)
    ○ = darker or similar
```

**Speed Optimization** (High-Speed Test):
```
First test pixels 1, 5, 9, 13 (NSEW positions)

       1                 If at least 3 of these 4
       ●                 are brighter/darker than p,
  13 ●   ● 5            continue testing all 16.
       ●
       9                 Otherwise, reject immediately.
                         (Very fast rejection!)
```

**Machine Learning Optimization**:
- Uses decision tree for fast classification
- Tests pixels in optimal order

**Parameters**:
- `threshold`: Intensity difference threshold
- `nonmaxSuppression`: Removes adjacent detections

**OpenCV**:
```python
fast = cv2.FastFeatureDetector_create(threshold=10)
keypoints = fast.detect(gray)
```

---

### 4. ORB (Oriented FAST and Rotated BRIEF)

**What it does**: Fast, rotation-invariant descriptor combining FAST and BRIEF.

**Components**:

1. **Keypoint Detection**: FAST with Harris score ranking

2. **Orientation Assignment**:
   ```
   θ = atan2(m₀₁, m₁₀)

   Where moments: mₚₓ = Σₓ,ᵧ xᵖyᵧ × I(x,y)
   ```

3. **rBRIEF Descriptor** (Rotated BRIEF):
   - Binary descriptor (256 bits = 32 bytes)
   - Tests pairs of pixels in rotated pattern
   - Steered by keypoint orientation

**Descriptor Computation**:
```
For each of 256 test pairs (pᵢ, qᵢ):
    bit[i] = 1 if I(pᵢ) < I(qᵢ) else 0
```

**OpenCV**:
```python
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(gray, None)
# descriptors: (N, 32) uint8 array
```

---

### 5. SIFT (Scale-Invariant Feature Transform)

**What it does**: Highly robust descriptor, invariant to scale and rotation.

**SIFT Pipeline Overview**:
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

**Algorithm Steps**:

1. **Scale-Space Construction**:
   ```
   L(x, y, σ) = G(x, y, σ) * I(x, y)
   ```
   Build Gaussian pyramid with multiple octaves and scales.

**Scale-Space Pyramid**:
```
Octave 1 (original)        Octave 2 (half size)       Octave 3 (quarter)
┌─────────────────┐        ┌───────────┐              ┌───────┐
│  σ = 1.6        │        │  σ = 3.2  │              │ σ=6.4 │
├─────────────────┤        ├───────────┤              ├───────┤
│  σ = 1.6×k      │        │  σ = 3.2k │              │  ...  │
├─────────────────┤        ├───────────┤              └───────┘
│  σ = 1.6×k²     │        │    ...    │
├─────────────────┤        └───────────┘
│  σ = 1.6×k³     │
├─────────────────┤        Each octave: image half
│  σ = 1.6×k⁴     │        the size of previous
└─────────────────┘
```

2. **DoG (Difference of Gaussians)**:
   ```
   D(x, y, σ) = L(x, y, kσ) - L(x, y, σ)
   ```
   Approximates Laplacian of Gaussian (LoG).

**DoG Computation**:
```
Gaussian Scale Space           Difference of Gaussians (DoG)

┌─────────────────┐
│    σ = kσ       │ ─────┐
└─────────────────┘      │     ┌─────────────────┐
                         ├───▶ │  DoG (σ, kσ)    │
┌─────────────────┐      │     └─────────────────┘
│    σ = σ        │ ─────┘
└─────────────────┘

   Subtract adjacent scales to find blob-like structures
```

3. **Keypoint Localization**:
   - Find extrema in 3×3×3 neighborhood
   - Subpixel refinement using Taylor expansion
   - Reject low contrast and edge responses

**Extrema Detection in 3×3×3**:
```
Scale above:    ○ ○ ○       Compare candidate ● with
                ○ ○ ○       26 neighbors (8 same scale,
                ○ ○ ○       9 above, 9 below)

Current scale:  ○ ○ ○       If ● is MAX or MIN among all
                ○ ● ○       26 neighbors → keypoint found
                ○ ○ ○

Scale below:    ○ ○ ○
                ○ ○ ○
                ○ ○ ○
```

4. **Orientation Assignment**:
   ```
   m(x,y) = √[(L(x+1,y) - L(x-1,y))² + (L(x,y+1) - L(x,y-1))²]
   θ(x,y) = atan2(L(x,y+1) - L(x,y-1), L(x+1,y) - L(x-1,y))
   ```
   Create 36-bin histogram of gradient orientations.

**Orientation Histogram**:
```
                    0°
                    │
         330°       │       30°
             ╲      │      ╱
              ╲     │     ╱
         270° ──────●────── 90°   Peak = dominant orientation
              ╱     │     ╲       (used to rotate descriptor)
             ╱      │      ╲
         210°       │       150°
                    │
                   180°

   36 bins × 10° each = full 360° coverage
```

5. **Descriptor Generation**:
   - 16×16 window around keypoint
   - Divide into 4×4 cells
   - 8-bin orientation histogram per cell
   - Result: 4×4×8 = 128-dimensional vector

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

**OpenCV**:
```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
# descriptors: (N, 128) float32 array
```

---

### 6. Feature Matching

**Feature Matching Concept**:
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
│   │  D₁, D₂...  │                      │  Dₐ, Dᵦ... │              │
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

#### Brute-Force Matcher

**Algorithm**:
```
For each descriptor in set A:
    Calculate distance to ALL descriptors in set B
    Return closest match(es)
```

**Brute-Force Matching**:
```
Descriptor A₁ compared to ALL in B:

  A₁ ──┬── dist(A₁, B₁) = 45
       ├── dist(A₁, B₂) = 12  ◀── Closest match!
       ├── dist(A₁, B₃) = 89
       ├── dist(A₁, B₄) = 156
       └── dist(A₁, B₅) = 67

  Match: A₁ ↔ B₂ (distance = 12)
```

**Distance Metrics**:
| Descriptor Type | Distance | Formula |
|-----------------|----------|---------|
| Float (SIFT) | L2 (Euclidean) | `√Σ(aᵢ - bᵢ)²` |
| Binary (ORB) | Hamming | Count of differing bits |

**Hamming Distance for Binary Descriptors**:
```
ORB descriptor (32 bytes = 256 bits)

A: 10110100 11001010 ...
B: 10100100 11011010 ...
   ──┬───── ──┬─────
     │        │
   XOR: 00010000 00010000 ...
                │
   Count 1s = Hamming distance

Fewer differing bits = more similar
```

**OpenCV**:
```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
```

#### FLANN (Fast Library for Approximate Nearest Neighbors)

**What it does**: Faster matching using tree-based indexing.

**KD-Tree for Fast Search**:
```
Instead of checking ALL descriptors (O(n)):

         ┌───────────────────────┐
         │       Root Node       │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    ┌─────────┐             ┌─────────┐
    │  Left   │             │  Right  │
    │ subtree │             │ subtree │
    └────┬────┘             └────┬────┘
         │                       │
    ┌────┴────┐             ┌────┴────┐
    ▼         ▼             ▼         ▼
 ┌─────┐  ┌─────┐       ┌─────┐  ┌─────┐
 │Leaf │  │Leaf │       │Leaf │  │Leaf │
 └─────┘  └─────┘       └─────┘  └─────┘

Quickly navigate to relevant region: O(log n)
```

**For binary descriptors** (ORB, BRISK):
```python
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                   table_number=6,
                   key_size=12,
                   multi_probe_level=1)
```

**For float descriptors** (SIFT):
```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
```

---

### 7. Lowe's Ratio Test

**What it does**: Filters ambiguous matches by comparing to second-best match.

**Formula**:
```
Keep match if: distance(best) < ratio × distance(second_best)
```

**Ratio Test Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Lowe's Ratio Test                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  GOOD MATCH (pass ratio test)      BAD MATCH (fail ratio test)     │
│                                                                     │
│  Descriptor A₁ → knnMatch(k=2)     Descriptor A₂ → knnMatch(k=2)   │
│                                                                     │
│  Best match:    B₃  dist = 15      Best match:    B₇  dist = 45    │
│  2nd best:      B₈  dist = 120     2nd best:      B₂  dist = 52    │
│                                                                     │
│  Ratio: 15/120 = 0.125             Ratio: 45/52 = 0.865            │
│                                                                     │
│  0.125 < 0.75  ✓ KEEP              0.865 > 0.75  ✗ REJECT          │
│                                                                     │
│  Clear winner = distinctive        No clear winner = ambiguous     │
│  feature                           feature                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Visual Intuition**:
```
GOOD: One clear match           BAD: Multiple similar matches

  A₁ ────●──────────────        A₂ ────●─●────────────
         │                            │ │
        B₃ (very close)              B₇ B₂ (similar distance)
         │
        B₈ (far away)               Ambiguous - which is correct?
```

Typical ratio: 0.7-0.8

**Intuition**: Good matches have a clearly best match; ambiguous features have multiple similar matches.

**OpenCV**:
```python
matches = bf.knnMatch(desc1, desc2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
```

---

### 8. Homography Estimation with RANSAC

**What Homography Does**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Homography Transformation                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Image 1 (Source)              Homography H          Image 2 (Dest)│
│   ┌─────────────┐               ─────────▶           ╱─────────────╲│
│   │   ●    ●    │                                   ╱   ●      ●    ││
│   │             │           Maps points            │                ││
│   │ ●    ●      │           from plane 1          │   ●    ●       ││
│   │             │           to plane 2             ╲               ╱│
│   └─────────────┘                                   ╲─────────────╱ │
│                                                                     │
│   Handles: rotation, translation, scale, shear, perspective        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Homography Matrix** maps points between planes:
```
[x']   [h₁₁ h₁₂ h₁₃] [x]
[y'] = [h₂₁ h₂₂ h₂₃] [y]
[1 ]   [h₃₁ h₃₂ h₃₃] [1]

x' = (h₁₁x + h₁₂y + h₁₃) / (h₃₁x + h₃₂y + h₃₃)
y' = (h₂₁x + h₂₂y + h₂₃) / (h₃₁x + h₃₂y + h₃₃)
```

**RANSAC Algorithm Visualization**:
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
│  ✗ = outliers                    (Bad H → few inliers)              │
│                                                                     │
│  After N iterations:                                                │
│  Keep H with MOST inliers                                           │
│  Recompute H using ALL inliers for best result                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**RANSAC Algorithm**:
```
1. Randomly select 4 point correspondences
2. Compute homography H from these 4 points
3. Count inliers (points that fit H within threshold)
4. Repeat steps 1-3 for N iterations
5. Return H with most inliers
6. Optionally: recompute H using all inliers
```

**Why 4 Points?**
```
Homography has 8 degrees of freedom
(9 elements - 1 for scale)

Each point correspondence gives 2 equations:
  x' = f(x, y, H)
  y' = g(x, y, H)

4 points × 2 equations = 8 equations
→ Enough to solve for 8 unknowns
```

**OpenCV**:
```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# mask indicates inliers
```

---

## Algorithm Comparison

| Algorithm | Speed | Descriptor Size | Type | Best For |
|-----------|-------|-----------------|------|----------|
| ORB | Fast | 32 bytes | Binary | Real-time, mobile |
| SIFT | Slow | 128 floats | Float | High accuracy |
| BRISK | Fast | 64 bytes | Binary | Scale-invariant |
| AKAZE | Medium | Variable | Binary | Deformable objects |
| FAST | Very Fast | N/A (detector only) | - | Real-time tracking |

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_corners.py` | Harris, Shi-Tomasi, FAST, subpixel accuracy |
| `02_descriptors.py` | ORB, SIFT, BRISK, AKAZE |
| `03_matching.py` | BF matcher, FLANN, ratio test, homography |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.cornerHarris()` | Harris corner detection |
| `cv2.goodFeaturesToTrack()` | Shi-Tomasi corners |
| `cv2.FastFeatureDetector_create()` | FAST detector |
| `cv2.ORB_create()` | ORB detector/descriptor |
| `cv2.SIFT_create()` | SIFT detector/descriptor |
| `cv2.BRISK_create()` | BRISK detector/descriptor |
| `cv2.AKAZE_create()` | AKAZE detector/descriptor |
| `cv2.BFMatcher()` | Brute-force matcher |
| `cv2.FlannBasedMatcher()` | FLANN matcher |
| `cv2.findHomography()` | Compute homography |
| `cv2.drawKeypoints()` | Visualize keypoints |
| `cv2.drawMatches()` | Visualize matches |

---

## Further Reading

- [Feature Detection Tutorial](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [SIFT Paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [ORB Paper](http://www.willowgarage.com/sites/default/files/orb_final.pdf)
