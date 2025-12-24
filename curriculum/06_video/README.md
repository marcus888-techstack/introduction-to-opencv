# Module 6: Video Analysis

Motion analysis and background modeling for video processing.

## Topics Covered

- Lucas-Kanade optical flow (sparse)
- Farneback optical flow (dense)
- Background subtraction (MOG2, KNN)
- Motion detection

---

## Algorithm Explanations

### 1. Optical Flow Concept

**What it does**: Estimates motion between consecutive frames.

**Optical Flow Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Optical Flow Concept                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Frame t                        Frame t+1                         │
│   ┌───────────────────┐          ┌───────────────────┐             │
│   │                   │          │                   │             │
│   │      ●            │          │           ●       │             │
│   │    (ball)         │  ──▶     │         (ball)    │             │
│   │                   │          │                   │             │
│   └───────────────────┘          └───────────────────┘             │
│                                                                     │
│   Optical Flow = Motion Vector                                      │
│                                                                     │
│         ●───────────────▶●                                          │
│        (x,y)            (x+dx, y+dy)                                │
│                                                                     │
│   Flow vector: (dx, dy) = displacement per frame                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Definition**: For each pixel (x, y), find displacement (dx, dy) such that:
```
I(x, y, t) = I(x + dx, y + dy, t + dt)
```

**Brightness Constancy Assumption**:
```
I(x + dx, y + dy, t + dt) = I(x, y, t)
```

**Taylor Expansion**:
```
I(x + dx, y + dy, t + dt) ≈ I + Iₓdx + Iᵧdy + Iₜdt
```

**Optical Flow Constraint Equation**:
```
Iₓu + Iᵧv + Iₜ = 0

Or: ∇I · v + Iₜ = 0
```

Where:
- `Iₓ, Iᵧ`: Spatial derivatives
- `Iₜ`: Temporal derivative
- `u = dx/dt`, `v = dy/dt`: Flow velocities

**Problem**: One equation, two unknowns → need additional constraints.

---

### 2. Lucas-Kanade Optical Flow (Sparse)

**What it does**: Tracks sparse feature points between frames.

**Sparse vs Dense Flow**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                  Sparse vs Dense Optical Flow                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   SPARSE (Lucas-Kanade)              DENSE (Farneback)             │
│   Track selected points              Compute flow for ALL pixels   │
│                                                                     │
│   ┌───────────────────┐              ┌───────────────────┐         │
│   │ ●→    ●→          │              │→→→→→→→→→→→→→→→→→ │         │
│   │                   │              │→→→→→→→→→→→→→→→→→ │         │
│   │    ●→       ●→    │              │→→→→→→→→→→→→→→→→→ │         │
│   │                   │              │→→→→→→→→→→→→→→→→→ │         │
│   │ ●→    ●→          │              │→→→→→→→→→→→→→→→→→ │         │
│   └───────────────────┘              └───────────────────┘         │
│                                                                     │
│   Fast, for tracking                 Slow, full motion field       │
│   specific features                  for visualization             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Additional Constraint**: Assume constant flow in local neighborhood.

For a window of n pixels:
```
[Iₓ₁  Iᵧ₁]       [Iₜ₁]
[Iₓ₂  Iᵧ₂] [u]   [Iₜ₂]
[  ⋮    ⋮ ] [v] = -[ ⋮ ]
[Iₓₙ  Iᵧₙ]       [Iₜₙ]

    A      v   =   b
```

**Least Squares Solution**:
```
v = (AᵀA)⁻¹Aᵀb

[u]   [Σ Iₓ²    Σ IₓIᵧ]⁻¹ [-Σ IₓIₜ]
[v] = [Σ IₓIᵧ   Σ Iᵧ² ]   [-Σ IᵧIₜ]
```

**Pyramidal Extension** (for large motions):
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Image Pyramid for Large Motion                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Problem: Large motion exceeds window size                         │
│   Solution: Compute at coarse scale, refine at fine scale          │
│                                                                     │
│   Level 2 (coarsest)      ┌─────┐                                   │
│   Large motion → small    │     │  Compute initial flow            │
│                           └─────┘                                   │
│                              │                                      │
│                              ▼                                      │
│   Level 1                ┌─────────┐                                │
│   Propagate & refine     │         │  Refine flow estimate         │
│                          └─────────┘                                │
│                              │                                      │
│                              ▼                                      │
│   Level 0 (finest)    ┌───────────────┐                             │
│   Final refinement    │               │  Final accurate flow       │
│                       └───────────────┘                             │
│                                                                     │
│   At each level: motion appears smaller (easier to track)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**OpenCV**:
```python
next_pts, status, error = cv2.calcOpticalFlowPyrLK(
    prev_gray, next_gray, prev_pts, None,
    winSize=(15, 15),    # Window size
    maxLevel=2,          # Pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
```

**Returns**:
- `next_pts`: New positions of tracked points
- `status`: 1 if flow found, 0 otherwise
- `error`: Tracking error

---

### 3. Farneback Optical Flow (Dense)

**What it does**: Computes flow for every pixel.

**Polynomial Expansion**:
Approximates neighborhood with quadratic polynomial:
```
f(x) ≈ xᵀAx + bᵀx + c
```

Where A is a symmetric matrix, b is a vector, c is a scalar.

**Displacement Estimation**:
For two frames with polynomial approximations:
```
f₁(x) ≈ xᵀA₁x + b₁ᵀx + c₁
f₂(x) ≈ xᵀA₂x + b₂ᵀx + c₂
```

Assuming f₂(x) = f₁(x - d):
```
d = -(A₁ + A₂)⁻¹ × (b₂ - b₁) / 2
```

**OpenCV**:
```python
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, next_gray, None,
    pyr_scale=0.5,      # Pyramid scale
    levels=3,           # Pyramid levels
    winsize=15,         # Averaging window
    iterations=3,       # Iterations per level
    poly_n=5,           # Polynomial neighborhood (5 or 7)
    poly_sigma=1.2,     # Gaussian smoothing
    flags=0
)
# Returns (H, W, 2) array: flow[y, x] = [dx, dy]
```

**Flow Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Flow Color Coding (HSV)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Direction → Hue (color)           Magnitude → Value (brightness) │
│                                                                     │
│              0° (Red)                                               │
│                 │                                                   │
│        315°    │    45°             Slow        Fast                │
│           ╲    │    ╱               (dark)      (bright)            │
│            ╲   │   ╱                                                │
│   270° ─────────────── 90°          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓              │
│   (Blue)    ╱   │   ╲    (Yellow)   ░░░░░░░░░░░░████████            │
│            ╱    │    ╲              ░░░░░░░██████████████           │
│        225°     │     135°                                          │
│                 │                                                   │
│              180° (Cyan)                                            │
│                                                                     │
│   Example: Moving right (0°) and fast = bright red                 │
│            Moving up (270°) and slow = dark blue                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# Convert to polar coordinates
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# HSV representation
hsv = np.zeros((h, w, 3), dtype=np.uint8)
hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
hsv[..., 1] = 255                       # Saturation = max
hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
```

---

### 4. Background Subtraction

**What it does**: Separates foreground (moving objects) from background (static).

**Background Subtraction Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Background Subtraction                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Frame          Background Model       Foreground Mask      │
│   ┌───────────────┐    ┌───────────────┐     ┌───────────────┐    │
│   │   ┌───┐       │    │               │     │   ┌───┐       │    │
│   │   │   │       │    │               │     │   │███│       │    │
│   │   │🚶‍♂️│       │ -  │   (empty      │  =  │   │███│       │    │
│   │   │   │       │    │    scene)     │     │   └───┘       │    │
│   │   └───┘       │    │               │     │               │    │
│   └───────────────┘    └───────────────┘     └───────────────┘    │
│                                                                     │
│   Current frame      Learned over time       White = foreground    │
│   with person        (no person)             Black = background    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### MOG2 (Mixture of Gaussians)

**Model**: Each pixel modeled as mixture of K Gaussians:
```
P(xₜ) = Σₖ wₖ × N(xₜ; μₖ, Σₖ)
```

**Gaussian Mixture Model per Pixel**:
```
┌─────────────────────────────────────────────────────────────────────┐
│               Pixel Modeled by Multiple Gaussians                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Pixel intensity histogram over time:                              │
│                                                                     │
│       ▲                                                             │
│       │    Gaussian 1         Gaussian 2                           │
│       │    (background:       (shadow:                              │
│       │     bright sky)        darker)                              │
│       │        ╱╲                ╱╲                                 │
│       │       ╱  ╲              ╱  ╲                                │
│   freq│      ╱    ╲            ╱    ╲                               │
│       │     ╱      ╲          ╱      ╲                              │
│       │    ╱        ╲        ╱        ╲                             │
│       │───────────────────────────────────────▶ intensity           │
│       0          100        150        200                          │
│                                                                     │
│   New pixel value:                                                  │
│   - Matches Gaussian → Background                                   │
│   - No match → Foreground (moving object)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Algorithm**:
```
1. For each new pixel value:
   a. Check which Gaussian matches (within 2.5σ)
   b. If match: update that Gaussian's parameters
   c. If no match: replace weakest Gaussian

2. Background: Gaussians with highest weights
3. Foreground: Pixels not matching background
```

**Update Rules**:
```
wₖ ← (1 - α)wₖ + α × Mₖ
μₖ ← (1 - ρ)μₖ + ρ × xₜ
σₖ² ← (1 - ρ)σₖ² + ρ × (xₜ - μₖ)²

Where:
  α = learning rate
  ρ = α / wₖ
  Mₖ = 1 if matched, 0 otherwise
```

**Shadow Detection**:
```
Shadow if: 0.5 < I/B < 1.0 and similar chromaticity
```

**OpenCV**:
```python
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,        # Frames used for background
    varThreshold=16,    # Squared Mahalanobis distance
    detectShadows=True  # Enable shadow detection
)
fg_mask = mog2.apply(frame)
# Returns: 0=background, 127=shadow, 255=foreground
```

#### KNN Background Subtractor

**Model**: Uses K nearest neighbors in sample history.

```
Background if: pixel is close to K samples in history
```

**OpenCV**:
```python
knn = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400,
    detectShadows=True
)
fg_mask = knn.apply(frame)
```

---

### 5. Motion Detection Pipeline

**Typical Workflow**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Motion Detection Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. Input Frame         2. BG Subtract        3. Morphology       │
│   ┌───────────────┐      ┌───────────────┐     ┌───────────────┐  │
│   │   ┌───┐       │      │   ┌───┐ noise │     │   ┌───┐       │  │
│   │   │🚗 │       │  ──▶ │   │███│ ·· ·  │ ──▶ │   │███│       │  │
│   │   └───┘       │      │   └───┘  ·    │     │   └───┘       │  │
│   └───────────────┘      └───────────────┘     └───────────────┘  │
│                                                  (noise removed)   │
│                                                                     │
│   4. Find Contours       5. Filter by Area     6. Draw Result     │
│   ┌───────────────┐      ┌───────────────┐     ┌───────────────┐  │
│   │   ┌───┐       │      │   ┌───┐       │     │   ┌───┐       │  │
│   │   │ ▢ │ small │  ──▶ │   │ ▢ │       │ ──▶ │   │🚗 │       │  │
│   │   └───┘  ·    │      │   └───┘       │     │   └───┘       │  │
│   └───────────────┘      └───────────────┘     └───────────────┘  │
│                           (ignore tiny)        (bounding box)      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
1. Apply background subtractor
2. Threshold/clean mask
3. Morphological opening (remove noise)
4. Morphological closing (fill holes)
5. Find contours
6. Filter by area
7. Draw bounding boxes
```

---

## Comparison

| Method | Type | Speed | Use Case |
|--------|------|-------|----------|
| Lucas-Kanade | Sparse | Fast | Feature tracking |
| Farneback | Dense | Medium | Full motion field |
| MOG2 | Background | Fast | Surveillance |
| KNN | Background | Fast | Complex backgrounds |

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_optical_flow.py` | Lucas-Kanade, Farneback, motion visualization |
| `02_background_subtraction.py` | MOG2, KNN, foreground detection |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.calcOpticalFlowPyrLK()` | Lucas-Kanade sparse flow |
| `cv2.calcOpticalFlowFarneback()` | Farneback dense flow |
| `cv2.createBackgroundSubtractorMOG2()` | Create MOG2 |
| `cv2.createBackgroundSubtractorKNN()` | Create KNN |
| `subtractor.apply(frame)` | Get foreground mask |
| `subtractor.getBackgroundImage()` | Get background model |

---

## Further Reading

- [Optical Flow Tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- [Lucas-Kanade Paper](https://cecas.clemson.edu/~stb/klt/lucas_bruce_kanade_tomasi_optical_flow.pdf)
