---
layout: default
title: "10: Photo"
parent: Modules
nav_order: 10
---

# Module 10: Computational Photography

Image enhancement, restoration, and artistic effects using computational photography techniques.

## Topics Covered

- Inpainting (image restoration)
- Non-local means denoising
- HDR imaging and tone mapping
- Seamless cloning
- Stylization effects

---

## Algorithm Explanations

### 1. Inpainting

**What it does**: Fills in missing or damaged regions using information from surrounding pixels.

**Inpainting Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Image Inpainting                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Original Image              Mask                 Inpainted       │
│                                                                     │
│   ┌───────────────┐       ┌───────────────┐    ┌───────────────┐  │
│   │   ▓▓▓▓▓▓▓    │       │               │    │   ▓▓▓▓▓▓▓    │  │
│   │ ▓▓▓▒▒▒▓▓▓   │       │     ████      │    │ ▓▓▓▓▓▓▓▓▓   │  │
│   │▓▓▓▓▒▒▓▓▓▓▓  │ mask  │    ██████     │ →  │▓▓▓▓▓▓▓▓▓▓▓  │  │
│   │ ▓▓▓▒▒▒▓▓▓   │ ───▶  │     ████      │    │ ▓▓▓▓▓▓▓▓▓   │  │
│   │   ▓▓▓▓▓▓▓    │       │               │    │   ▓▓▓▓▓▓▓    │  │
│   └───────────────┘       └───────────────┘    └───────────────┘  │
│                                                                     │
│   ▒ = damaged/missing     █ = mask (white)    Restored using      │
│       area                    inpaint here     surrounding info   │
│                                                                     │
│   Use Cases:                                                        │
│   • Remove objects (power lines, people)                           │
│   • Restore old/damaged photos                                     │
│   • Remove watermarks/text                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Inpainting Methods Comparison**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    NS vs Telea Methods                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Navier-Stokes (INPAINT_NS)         Telea (INPAINT_TELEA)        │
│                                                                     │
│   ┌───────────────┐                  ┌───────────────┐             │
│   │ ──→ ──→ ──→   │                  │ ↘ → → → ↙   │             │
│   │ ──→ ??? ──→   │   Fluid flow     │ ↓ ??? ↑   │   Fast march  │
│   │ ──→ ──→ ──→   │   propagation    │ ↗ ← ← ← ↖   │   from edge  │
│   └───────────────┘                  └───────────────┘             │
│                                                                     │
│   • Better for large regions         • Faster                      │
│   • Follows isophotes                • Good for small regions      │
│   • Smoother results                 • Weighted average            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Navier-Stokes Method (`INPAINT_NS`)

Based on fluid dynamics equations for smooth propagation.

**Algorithm**:
```
1. Propagate isophote lines (equal intensity contours) into the damaged region
2. Use Navier-Stokes equations for fluid flow:
   ∂I/∂t + ∇I · ∇(ΔI) = 0

3. Iterate until convergence
```

**Isophote Propagation**:
```
∇I = gradient (direction of fastest intensity change)
ΔI = Laplacian (smoothness)

Isophotes flow perpendicular to gradient
```

#### Telea's Method (`INPAINT_TELEA`)

Fast marching method that fills from boundary inward.

**Algorithm**:
```
1. Start from region boundary
2. For each pixel to be filled:
   a. Use weighted average of known neighbors
   b. Weights based on:
      - Distance to pixel
      - Boundary proximity
      - Level line direction

   I(p) = Σₓ w(q) × [I(q) + ∇I(q) · (p - q)] / Σₓ w(q)
```

**Weight Function**:
```
w(q) = dir(p,q) × dst(p,q) × lev(p,q)

Where:
  dir = directional component (gradient alignment)
  dst = geometric distance factor
  lev = level line factor
```

**OpenCV**:
```python
# Create mask (white = regions to inpaint)
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.circle(mask, (x, y), radius, 255, -1)

# Inpaint
result_ns = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
result_telea = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
```

**Parameter**:
- `inpaintRadius`: Neighborhood radius for each point being inpainted

---

### 2. Non-Local Means Denoising

**What it does**: Removes noise while preserving edges by averaging similar patches across the image.

**Non-Local Means Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Non-Local Means Denoising                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Traditional (Local)              Non-Local Means                 │
│                                                                     │
│   ┌─────────────────┐              ┌─────────────────┐             │
│   │     [P]         │              │ [P]             │ Similar    │
│   │    ╱│╲          │              │                 │ patches   │
│   │   average of    │              │        [S1]     │ across    │
│   │   neighbors     │              │  [S2]     [S3]  │ whole     │
│   │                 │              │       [S4]      │ image!    │
│   └─────────────────┘              └─────────────────┘             │
│                                                                     │
│   Only uses pixels            Searches for similar patches         │
│   right next to P             anywhere in the search window        │
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │   For each pixel P:                                        │    │
│   │   1. Extract patch around P                                │    │
│   │   2. Search for similar patches in window                  │    │
│   │   3. Compute weighted average (higher weight = more similar)│    │
│   │   4. Result = weighted blend of all similar patches        │    │
│   └───────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Patch Similarity**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Patch Comparison                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Template Patch (at P)      Search Window                         │
│   ┌───────┐                  ┌─────────────────────┐               │
│   │ 5 5 5 │                  │   [S1]         [S2] │               │
│   │ 5 X 5 │                  │ 5 5 5       3 3 3   │ weight=0.9   │
│   │ 5 5 5 │                  │ 5 ? 5       3 ? 3   │ weight=0.1   │
│   └───────┘                  │ 5 5 5       3 3 3   │               │
│                              │       [S3]          │               │
│   templateWindowSize=7       │     5 5 5           │ weight=0.85  │
│                              │     5 ? 5           │               │
│   searchWindowSize=21        │     5 5 5           │               │
│                              └─────────────────────┘               │
│                                                                     │
│   Final value = Σ(weight × patch_center) / Σ(weight)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Similar patches exist throughout the image, not just locally.

**Algorithm**:
```
For each pixel p:
    NL[u](p) = Σₓ w(p, q) × u(q) / Σₓ w(p, q)

Where:
  u(q) = pixel value at q
  w(p, q) = similarity weight between patches at p and q
```

**Weight Calculation**:
```
w(p, q) = exp(-||P(p) - P(q)||² / h²)

Where:
  P(p) = patch centered at p
  h = filtering parameter (denoising strength)
  ||.||² = weighted Euclidean distance
```

**Patch Distance**:
```
d(p, q) = (1/|N|) × Σᵢ∈N (P(p)ᵢ - P(q)ᵢ)²

N = patch neighborhood
```

**OpenCV**:
```python
# Grayscale
denoised = cv2.fastNlMeansDenoising(
    src,
    None,
    h=10,                # Filter strength
    templateWindowSize=7, # Patch size (odd)
    searchWindowSize=21   # Search area (odd)
)

# Color
denoised_color = cv2.fastNlMeansDenoisingColored(
    src,
    None,
    h=10,              # Luminance strength
    hForColorComponents=10,  # Color strength
    templateWindowSize=7,
    searchWindowSize=21
)

# Video (multiple frames)
denoised_video = cv2.fastNlMeansDenoisingMulti(
    srcImgs,           # List of frames
    imgToDenoiseIndex, # Index of frame to denoise
    temporalWindowSize # Number of frames to use
)
```

**Parameter Tuning**:
| Parameter | Effect of Increase |
|-----------|-------------------|
| `h` | More smoothing, may lose details |
| `templateWindowSize` | Larger patches, slower |
| `searchWindowSize` | Larger search, slower, better |

---

### 3. HDR Imaging

**What it does**: Combines multiple exposures to capture full dynamic range.

**HDR Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    HDR: High Dynamic Range                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Real World:              Camera Sensor:           HDR Goal:      │
│   Dynamic Range            Limited Range            Capture All    │
│                                                                     │
│   ████████████████         ███░░░░░░░░░░           ████████████████│
│   Bright sky               Clipped!                Good sky        │
│                                                                     │
│   ░░░░░░░░░░░░░░░░         ░░░░░░░░░░░░░           ░░░░░░░░░░░░░░░░│
│   Dark shadows             Too dark                Good shadows    │
│                                                                     │
│   Multiple Exposures → Merge → HDR → Tonemap → Displayable        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**HDR Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    HDR Processing Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Short Exposure         Medium Exposure        Long Exposure      │
│   (bright areas OK)      (midtones OK)          (shadows OK)       │
│                                                                     │
│   ┌─────────┐            ┌─────────┐            ┌─────────┐        │
│   │ ████░░░ │            │ ███████ │            │ ░░░████ │        │
│   │ ████░░░ │            │ ███░███ │            │ ░░░░░██ │        │
│   │ ████░░░ │            │ ███████ │            │ ░░░████ │        │
│   └────┬────┘            └────┬────┘            └────┬────┘        │
│        │                      │                      │              │
│        └──────────────────────┼──────────────────────┘              │
│                               │                                     │
│                               ▼                                     │
│                    ┌─────────────────┐                             │
│                    │   HDR Merge     │  createMergeDebevec()       │
│                    │ (32-bit float)  │  createMergeMertens()       │
│                    └────────┬────────┘                             │
│                             │                                       │
│                             ▼                                       │
│                    ┌─────────────────┐                             │
│                    │    Tonemap      │  createTonemap()            │
│                    │  (compress to   │  createTonemapDrago()       │
│                    │   8-bit LDR)    │  createTonemapReinhard()    │
│                    └────────┬────────┘                             │
│                             │                                       │
│                             ▼                                       │
│                    ┌─────────────────┐                             │
│                    │ Final Image     │                             │
│                    │ (displayable)   │                             │
│                    └─────────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Exposure Fusion Pipeline

```
1. Capture: Multiple exposures of same scene
2. Align: Compensate for camera motion
3. Merge: Combine into HDR image
4. Tonemap: Compress to displayable range
```

#### Camera Response Function

**Recovering Radiance**:
```
Z = f(E × Δt)

Where:
  Z = pixel value
  f = camera response function
  E = scene radiance
  Δt = exposure time

Inverse: E = f⁻¹(Z) / Δt
```

**Debevec's Method**:
```
g(Z) = ln(E) + ln(Δt)

Solve for g using multiple exposures:
Minimize: Σᵢ Σⱼ [g(Zᵢⱼ) - ln(Eᵢ) - ln(Δtⱼ)]² + λ × Σᵢ g''(z)²
```

#### HDR Merge Methods

**Debevec Merge**:
```python
merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, times=exposure_times)
```

**Robertson Merge**:
```python
merge_robertson = cv2.createMergeRobertson()
hdr = merge_robertson.process(images, times=exposure_times)
```

**Mertens Fusion** (no HDR, direct fusion):
```python
merge_mertens = cv2.createMergeMertens()
fusion = merge_mertens.process(images)  # No exposure times needed
```

#### Tone Mapping

**Drago Tonemap**:
```
L_d = L_max × log(1 + L_w) / log(1 + L_max)
```

**Reinhard Tonemap**:
```
L_d = L_w / (1 + L_w)

With key value:
L_d = (key / L_avg) × L_w / (1 + (key / L_avg) × L_w)
```

**OpenCV Tonemappers**:
```python
# Simple gamma
tonemap = cv2.createTonemap(gamma=2.2)

# Drago
tonemap_drago = cv2.createTonemapDrago(gamma=2.2, saturation=1.0)

# Reinhard
tonemap_reinhard = cv2.createTonemapReinhard(gamma=2.2, intensity=0, light_adapt=0, color_adapt=0)

# Mantiuk
tonemap_mantiuk = cv2.createTonemapMantiuk(gamma=2.2, scale=0.85, saturation=1.0)

ldr = tonemap.process(hdr)
```

---

### 4. Seamless Cloning

**What it does**: Blends source object into destination, matching colors and lighting.

**Seamless Clone Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Seamless Cloning (Poisson Blending)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Source Image         Destination Image      Result               │
│                                                                     │
│   ┌───────────┐        ┌───────────────┐      ┌───────────────┐   │
│   │   ┌───┐   │        │ ░░░░░░░░░░░░░ │      │ ░░░░░░░░░░░░░ │   │
│   │   │🌸│   │   +    │ ░░░░░░░░░░░░░ │  =   │ ░░░┌───┐░░░░░ │   │
│   │   └───┘   │        │ ░░░░░░░░░░░░░ │      │ ░░░│🌸│░░░░░ │   │
│   └───────────┘        │ ░░░░░░░░░░░░░ │      │ ░░░└───┘░░░░░ │   │
│        +               └───────────────┘      └───────────────┘   │
│   ┌───────────┐                                                    │
│   │   ███     │  Mask                                              │
│   │   ███     │                                                    │
│   │   ███     │                                                    │
│   └───────────┘                                                    │
│                                                                     │
│   Magic: Colors and lighting automatically blend to match!         │
│   No visible seams at the boundary                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Copy-Paste vs Seamless Clone**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Why Seamless Cloning?                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Simple Copy-Paste               Seamless Clone                   │
│                                                                     │
│   ┌───────────────────┐          ┌───────────────────┐             │
│   │ ░░░░░░░░░░░░░░░░░ │          │ ░░░░░░░░░░░░░░░░░ │             │
│   │ ░░░┌───────┐░░░░░ │          │ ░░░▒▒▒▒▒▒▒▒▒░░░░ │             │
│   │ ░░░│███████│░░░░░ │          │ ░░░░▒▒▒▒▒▒▒░░░░░ │             │
│   │ ░░░│███████│░░░░░ │          │ ░░░░░▒▒▒▒▒░░░░░░ │             │
│   │ ░░░└───────┘░░░░░ │          │ ░░░░░░░░░░░░░░░░ │             │
│   │ ░░░░░░░░░░░░░░░░░ │          │ ░░░░░░░░░░░░░░░░░ │             │
│   └───────────────────┘          └───────────────────┘             │
│                                                                     │
│   Visible edge!                  Smooth transition!                │
│   Color mismatch                 Colors blend naturally            │
│                                                                     │
│   Poisson blending preserves gradients (edges) from source        │
│   while matching boundary colors from destination                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Poisson Equation**:
```
Minimize: ∫∫_Ω |∇f - v|²

Subject to: f|∂Ω = f*|∂Ω

Where:
  f = output image
  v = guidance field (source gradients)
  Ω = cloning region
  ∂Ω = region boundary
  f* = destination image
```

**Discrete Laplacian**:
```
Δf(p) = Σ_q∈N(p) (f(q) - f(p))

For each interior pixel p:
|N(p)| × f(p) - Σ_q∈N(p) f(q) = Σ_q∈N(p) v_pq

Where v_pq = g(p) - g(q) (source gradients)
```

**Cloning Modes**:

| Mode | Description |
|------|-------------|
| `NORMAL_CLONE` | Transfers texture and color from source |
| `MIXED_CLONE` | Uses stronger gradient from either source or dest |
| `MONOCHROME_TRANSFER` | Transfers texture only (grayscale) |

**Mixed Clone Gradient**:
```
v_pq = {
  g(p) - g(q)       if |g(p) - g(q)| > |f*(p) - f*(q)|
  f*(p) - f*(q)     otherwise
}
```

**OpenCV**:
```python
# Create mask (white = region to clone)
mask = np.zeros(source.shape[:2], dtype=np.uint8)
cv2.circle(mask, (cx, cy), radius, 255, -1)

# Clone center point in destination
center = (dest_x, dest_y)

# Seamless clone
result = cv2.seamlessClone(
    source,
    destination,
    mask,
    center,
    cv2.NORMAL_CLONE  # or MIXED_CLONE, MONOCHROME_TRANSFER
)
```

---

### 5. Stylization Effects

**Edge-Preserving Filtering**:
```
Smooths image while preserving strong edges.

σ_s = spatial extent (larger = more smoothing)
σ_r = color/range extent (larger = less edge preservation)
```

**OpenCV Stylization Functions**:

```python
# Artistic stylization
stylized = cv2.stylization(src, sigma_s=60, sigma_r=0.45)

# Pencil sketch
gray_sketch, color_sketch = cv2.pencilSketch(
    src,
    sigma_s=60,
    sigma_r=0.07,
    shade_factor=0.05
)

# Detail enhancement
enhanced = cv2.detailEnhance(src, sigma_s=10, sigma_r=0.15)

# Edge-preserving filter
filtered = cv2.edgePreservingFilter(
    src,
    flags=cv2.RECURS_FILTER,  # or NORMCONV_FILTER
    sigma_s=60,
    sigma_r=0.4
)
```

**Parameters**:
| Parameter | Effect |
|-----------|--------|
| `sigma_s` | Spatial smoothing (0-200) |
| `sigma_r` | Color smoothing (0-1) |
| `shade_factor` | Pencil shading intensity |

---

## Comparison

| Technique | Purpose | Speed | Use Case |
|-----------|---------|-------|----------|
| Inpainting NS | Restoration | Medium | Smooth regions |
| Inpainting Telea | Restoration | Fast | General |
| NL Means | Denoising | Slow | High quality |
| HDR Merge | Dynamic range | Medium | High contrast scenes |
| Seamless Clone | Compositing | Medium | Object insertion |
| Stylization | Artistic | Fast | Visual effects |

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_photo_basics.py` | Inpainting, denoising, HDR, seamless cloning, stylization |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.inpaint()` | Restore damaged regions |
| `cv2.fastNlMeansDenoising()` | Denoise grayscale |
| `cv2.fastNlMeansDenoisingColored()` | Denoise color |
| `cv2.createMergeDebevec()` | HDR merge |
| `cv2.createMergeMertens()` | Exposure fusion |
| `cv2.createTonemap()` | HDR tone mapping |
| `cv2.seamlessClone()` | Poisson blending |
| `cv2.stylization()` | Artistic effect |
| `cv2.pencilSketch()` | Sketch effect |
| `cv2.detailEnhance()` | Enhance details |
| `cv2.edgePreservingFilter()` | Smooth preserving edges |

---

## Further Reading

- [Inpainting Tutorial](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html)
- [Denoising Tutorial](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html)
- [HDR Tutorial](https://docs.opencv.org/4.x/d2/df0/tutorial_py_hdr.html)
- [Poisson Blending Paper](http://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
