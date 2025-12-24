---
layout: default
title: "06: Video Analysis"
parent: Modules
nav_order: 6
permalink: /modules/06-video
---

# Module 6: Video Analysis
{: .fs-9 }

Motion analysis and background modeling for video processing.
{: .fs-6 .fw-300 }

---

## Topics Covered

- Lucas-Kanade optical flow (sparse)
- Farneback optical flow (dense)
- Background subtraction (MOG2, KNN)
- Motion detection

---

## Algorithm Explanations

### 1. Optical Flow Concept

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
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2. Sparse vs Dense Optical Flow

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

---

### 3. Image Pyramid for Large Motion

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Image Pyramid for Large Motion                   │
├─────────────────────────────────────────────────────────────────────┤
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
└─────────────────────────────────────────────────────────────────────┘
```

---

### 4. Flow Color Coding

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
│   270° ─────────────── 90°                                          │
│   (Blue)    ╱   │   ╲    (Yellow)                                   │
│            ╱    │    ╲                                              │
│        225°     │     135°                                          │
│                 │                                                   │
│              180° (Cyan)                                            │
│                                                                     │
│   Example: Moving right (0°) and fast = bright red                 │
│            Moving up (270°) and slow = dark blue                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5. Background Subtraction

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Background Subtraction                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Frame          Background Model       Foreground Mask      │
│   ┌───────────────┐    ┌───────────────┐     ┌───────────────┐    │
│   │   ┌───┐       │    │               │     │   ┌───┐       │    │
│   │   │   │       │    │               │     │   │███│       │    │
│   │   │ P │       │ -  │   (empty      │  =  │   │███│       │    │
│   │   │   │       │    │    scene)     │     │   └───┘       │    │
│   │   └───┘       │    │               │     │               │    │
│   └───────────────┘    └───────────────┘     └───────────────┘    │
│                                                                     │
│   Current frame      Learned over time       White = foreground    │
│   with person        (no person)             Black = background    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 6. Gaussian Mixture Model per Pixel

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

---

## Comparison

| Method | Type | Speed | Use Case |
|:-------|:-----|:------|:---------|
| Lucas-Kanade | Sparse | Fast | Feature tracking |
| Farneback | Dense | Medium | Full motion field |
| MOG2 | Background | Fast | Surveillance |
| KNN | Background | Fast | Complex backgrounds |

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_optical_flow.py` | Lucas-Kanade, Farneback, motion visualization |
| `02_background_subtraction.py` | MOG2, KNN, foreground detection |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
| `cv2.calcOpticalFlowPyrLK()` | Lucas-Kanade sparse flow |
| `cv2.calcOpticalFlowFarneback()` | Farneback dense flow |
| `cv2.createBackgroundSubtractorMOG2()` | Create MOG2 |
| `cv2.createBackgroundSubtractorKNN()` | Create KNN |
| `subtractor.apply(frame)` | Get foreground mask |

---

## Further Reading

- [Optical Flow Tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
