---
layout: default
title: "02: Image Processing"
parent: Modules
nav_order: 2
permalink: /modules/02-imgproc
---

# Module 2: Image Processing

Comprehensive image processing operations including filtering, morphology, edge detection, and histogram analysis.

## Topics Covered

- Image filtering (blur, sharpen)
- Morphological operations
- Edge detection
- Contour detection and analysis
- Color spaces and histograms

---

## Algorithm Explanations

### 1. Convolution (Filtering)

**What it does**: Applies a kernel (filter) to an image by sliding it over each pixel.

**Mathematical Formula**:
```
G(x,y) = Σᵢ Σⱼ K(i,j) × I(x+i, y+j)
```

Where:
- `G(x,y)`: output pixel
- `K`: kernel (filter matrix)
- `I`: input image
- Sum over kernel dimensions

**Convolution Process Visualization**:
```
         Input Image                    Kernel (3×3)
┌───┬───┬───┬───┬───┬───┐          ┌────┬────┬────┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │          │ k00│ k01│ k02│
├───┼───┼───┼───┼───┼───┤          ├────┼────┼────┤
│ 7 │ 8 │ 9 │10 │11 │12 │          │ k10│ k11│ k12│
├───┼───┼───┼───┼───┼───┤          ├────┼────┼────┤
│13 │14 │15 │16 │17 │18 │          │ k20│ k21│ k22│
├───┼───┼───┼───┼───┼───┤          └────┴────┴────┘
│19 │20 │21 │22 │23 │24 │
└───┴───┴───┴───┴───┴───┘

Step 1: Position kernel    Step 2: Multiply & Sum    Step 3: Slide kernel
┌───────────────┐          Output = k00×1 + k01×2    ┌───────────────┐
│┌───┬───┬───┐  │               + k02×3 + k10×7     │  ┌───┬───┬───┐│
││ 1 │ 2 │ 3 │  │               + k11×8 + k12×9     │  │ 2 │ 3 │ 4 ││
│├───┼───┼───┤  │               + k20×13 + k21×14   │  ├───┼───┼───┤│
││ 7 │ 8 │ 9 │  │               + k22×15            │  │ 8 │ 9 │10 ││
│├───┼───┼───┤  │                    │              │  ├───┼───┼───┤│
││13 │14 │15 │  │                    ▼              │  │14 │15 │16 ││
│└───┴───┴───┘  │             Output[1,1]           │  └───┴───┴───┘│
└───────────────┘                                   └───────────────┘
                                                        Repeat...
```

---

### 2. Box Filter (Averaging)

**What it does**: Replaces each pixel with the average of its neighborhood.

**Kernel** (3×3):
```
     1  [1 1 1]
K = ─── [1 1 1]
     9  [1 1 1]
```

**Effect Visualization**:
```
Before (sharp edge):              After (blurred edge):
┌────────────────────┐            ┌────────────────────┐
│████████│          │            │████████▓▓░░        │
│████████│          │     ──►    │████████▓▓░░        │
│████████│          │            │████████▓▓░░        │
└────────────────────┘            └────────────────────┘
                                   Gradual transition
```

**OpenCV**: `cv2.blur(src, (ksize, ksize))`

---

### 3. Gaussian Blur

**What it does**: Weighted average where center pixels have more influence (bell curve).

**Gaussian Kernel Formula**:
```
G(x,y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)
```

**Gaussian Bell Curve (1D cross-section)**:
```
Weight
  ▲
1.0│        ╭─────╮
   │       ╱       ╲
   │      ╱         ╲
0.5│     ╱           ╲
   │    ╱             ╲
   │   ╱               ╲
0.0│──╱─────────────────╲──►
      -3σ  -σ  0  σ  3σ   Distance from center
         Center pixel
         has most weight
```

**5×5 Gaussian Kernel** (σ ≈ 1):
```
      1  [ 1  4  7  4  1]      Visualized:
K = ─── [ 4 16 26 16  4]       ░ = low weight
    273 [ 7 26 41 26  7]       ▓ = medium
        [ 4 16 26 16  4]       █ = high weight
        [ 1  4  7  4  1]
                               [░ ░ ▓ ░ ░]
                               [░ ▓ █ ▓ ░]
                               [▓ █ █ █ ▓]
                               [░ ▓ █ ▓ ░]
                               [░ ░ ▓ ░ ░]
```

**OpenCV**: `cv2.GaussianBlur(src, (ksize, ksize), sigma)`

---

### 4. Median Filter

**What it does**: Replaces each pixel with the median of its neighborhood.

**Algorithm**:
```
Neighborhood:           Sorted values:         Result:
┌─────┬─────┬─────┐
│ 120 │  35 │ 115 │    [35, 98, 105, 110,     Center pixel
├─────┼─────┼─────┤     115, 115, 118, 120,   replaced with
│ 105 │[255]│  98 │     255]                  median = 115
├─────┼─────┼─────┤           ↑
│ 110 │ 115 │ 118 │       middle value
└─────┴─────┴─────┘

Spike noise (255) is completely removed!
```

**Salt-and-Pepper Noise Removal**:
```
    With Noise              Gaussian Blur           Median Filter
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ ▪   ▪       ▪   │     │ still visible   │     │                 │
│   ▪     ▪       │ ──► │ blurred spots   │     │  Clean image!   │
│     ▪   ▪   ▪   │     │ everywhere      │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                           (poor)                  (excellent)
```

**OpenCV**: `cv2.medianBlur(src, ksize)`

---

### 5. Bilateral Filter

**What it does**: Smooths while preserving edges using both spatial and intensity distance.

**How it works**:
```
                    Spatial Weight              Intensity Weight
                    (distance-based)            (similarity-based)
                          ×                           ×

   Near pixels         Similar intensity
   weighted more        weighted more
        │                    │
        ▼                    ▼
    ┌───────┐            ┌───────┐
    │ █ █ █ │            │ █ ░ █ │
    │ █ ● █ │     ×      │ █ ░ █ │     = Final weight
    │ █ █ █ │            │ █ ░ █ │
    └───────┘            └───────┘
   Distance from         Different intensity
   center pixel          = low weight

Edge Preservation:
┌────────────────────┐         ┌────────────────────┐
│███████████│░░░░░░░░│   ──►   │███████████│░░░░░░░░│
│███████████│░░░░░░░░│         │███████████│░░░░░░░░│
│███████████│░░░░░░░░│         │███████████│░░░░░░░░│
└────────────────────┘         └────────────────────┘
   Sharp edge                   Edge preserved!
                               (only similar pixels averaged)
```

**OpenCV**: `cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)`

---

### 6. Sharpening

**What it does**: Enhances edges and fine details.

**Sharpening Kernel**:
```
      [ 0 -1  0]
K =   [-1  5 -1]    Center = 5, neighbors = -1
      [ 0 -1  0]    Sum = 1 (maintains brightness)
```

**How sharpening works**:
```
Original:        Blurred:         Original - Blurred = Edges:
┌───────────┐    ┌───────────┐    ┌───────────┐
│░░░░████░░░│    │░░░▓▓▓▓▓░░░│    │   ▓▓  ▓▓  │
│░░░░████░░░│ -  │░░░▓▓▓▓▓░░░│ =  │   ▓▓  ▓▓  │  (edge detail)
│░░░░████░░░│    │░░░▓▓▓▓▓░░░│    │   ▓▓  ▓▓  │
└───────────┘    └───────────┘    └───────────┘

Original + α × Edges = Sharpened:
┌───────────┐
│░░░░████░░░│  +  edges  =  Enhanced edges!
│░░░░████░░░│
└───────────┘
```

---

### 7. Morphological Operations

#### Structuring Element (Kernel)
```
Rectangle:    Ellipse:      Cross:       Diamond:
[1 1 1]      [0 1 0]      [0 1 0]      [0 0 1 0 0]
[1 1 1]      [1 1 1]      [1 1 1]      [0 1 1 1 0]
[1 1 1]      [0 1 0]      [0 1 0]      [1 1 1 1 1]
                                       [0 1 1 1 0]
                                       [0 0 1 0 0]
```

#### Erosion
**What it does**: Shrinks white regions, removes small noise.

```
Original:              Erosion:              Effect:
┌─────────────────┐    ┌─────────────────┐
│  ████████████   │    │    ████████     │   - Shrinks objects
│  ████████████   │──► │    ████████     │   - Removes small spots
│  ████████████   │    │    ████████     │   - Separates touching objects
│    ▪   ▪  ▪     │    │                 │
└─────────────────┘    └─────────────────┘
  Small spots removed!
```

#### Dilation
**What it does**: Expands white regions, fills small holes.

```
Original:              Dilation:             Effect:
┌─────────────────┐    ┌─────────────────┐
│    ████████     │    │  ████████████   │   - Expands objects
│    █▪█▪████     │──► │  ████████████   │   - Fills small holes
│    ████████     │    │  ████████████   │   - Connects nearby objects
└─────────────────┘    └─────────────────┘
  Holes filled!
```

#### Opening = Erosion → Dilation
```
Original              Erode                 Dilate
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  ████████████   │   │    ████████     │   │  ████████████   │
│  ████████████   │──►│    ████████     │──►│  ████████████   │
│    ▪  ▪   ▪     │   │                 │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                      Noise removed          Shape restored
```

#### Closing = Dilation → Erosion
```
Original              Dilate                Erode
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│    ████████     │   │  ████████████   │   │    ████████     │
│    █▪█▪████     │──►│  ████████████   │──►│    ████████     │
│    ████████     │   │  ████████████   │   │    ████████     │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                      Holes filled           Shape restored
```

#### Morphological Gradient = Dilation - Erosion
```
Original              Dilation - Erosion = Gradient (Edge)
┌─────────────────┐   ┌─────────────────┐
│                 │   │                 │
│    ████████     │──►│    ┌──────┐     │  Only boundaries
│    ████████     │   │    │      │     │  are visible!
│    ████████     │   │    └──────┘     │
│                 │   │                 │
└─────────────────┘   └─────────────────┘
```

---

### 8. Edge Detection

#### Sobel Operator

**What it does**: Computes first derivative (gradient) in X or Y direction.

**Kernels**:
```
Sobel X (vertical edges):     Sobel Y (horizontal edges):
    [-1  0  1]                    [-1 -2 -1]
    [-2  0  2]                    [ 0  0  0]
    [-1  0  1]                    [ 1  2  1]
```

**Direction Detection**:
```
            Sobel X detects │      Sobel Y detects ───
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    │   Detects:  █ │ ░     │   Detects:  ████      │
    │             █ │ ░     │             ────      │
    │             █ │ ░     │             ░░░░      │
    │                       │                       │
    └───────────────────────┴───────────────────────┘
```

**Gradient Magnitude and Direction**:
```
                    Gx                 Gy
                    ──►                ──►

              G = √(Gx² + Gy²)     θ = arctan(Gy/Gx)
                     │                    │
                     ▼                    ▼
               Edge strength         Edge direction
```

#### Canny Edge Detection Pipeline

```
Step 1: Gaussian Blur          Step 2: Gradient (Sobel)
┌─────────────────┐            ┌─────────────────┐
│ Noisy image     │    ──►     │ Gradient magnitude│
│ with edges      │            │ and direction    │
└─────────────────┘            └─────────────────┘
        │                               │
        ▼                               ▼
Step 3: Non-Max Suppression    Step 4: Double Threshold
┌─────────────────┐            ┌─────────────────┐
│ Thin edges to   │    ──►     │ Strong / Weak   │
│ 1-pixel width   │            │ edge pixels     │
└─────────────────┘            └─────────────────┘
        │                               │
        ▼                               ▼
Step 5: Hysteresis            Final Output
┌─────────────────┐            ┌─────────────────┐
│ Connect weak to │    ──►     │ Clean, thin     │
│ strong edges    │            │ edge map        │
└─────────────────┘            └─────────────────┘
```

**Non-Maximum Suppression Detail**:
```
    Gradient direction: →

    ┌───┬───┬───┐         Only keep if pixel is
    │ a │ b │ c │         maximum along gradient
    ├───┼───┼───┤         direction:
    │ d │[P]│ e │  ──►    Keep P if P > d AND P > e
    ├───┼───┼───┤
    │ f │ g │ h │
    └───┴───┴───┘
```

**Double Threshold**:
```
Pixel Intensity
      ▲
      │      ┌──────────────────── Strong Edge (keep)
 High │......│......................
      │      │
      │      │   Weak Edge (maybe keep)
  Low │------│----------------------
      │      │
      │      │   Non-edge (discard)
    0 └──────┴─────────────────────►
                   Pixels
```

---

### 9. Contour Detection

**What it does**: Finds continuous curves bounding objects.

**Contour Hierarchy**:
```
                    Image with nested contours:
    ┌─────────────────────────────────────────┐
    │  ┌─────────────────────────────────┐    │
    │  │ Contour 0 (outer)               │    │
    │  │   ┌───────────────────────┐     │    │
    │  │   │ Contour 1 (hole)      │     │    │
    │  │   │   ┌───────────────┐   │     │    │
    │  │   │   │ Contour 2     │   │     │    │
    │  │   │   │ (nested)      │   │     │    │
    │  │   │   └───────────────┘   │     │    │
    │  │   └───────────────────────┘     │    │
    │  └─────────────────────────────────┘    │
    └─────────────────────────────────────────┘

    Hierarchy: [Next, Previous, First_Child, Parent]
    Contour 0: [-1, -1, 1, -1]  ◄── Has child 1
    Contour 1: [-1, -1, 2, 0]   ◄── Parent is 0, has child 2
    Contour 2: [-1, -1, -1, 1]  ◄── Parent is 1, no children
```

**Contour Approximation (Douglas-Peucker)**:
```
Original contour:           Approximated:
    ●───●───●                   ●─────●
   /         \                 /       \
  ●           ●      ──►      ●         ●
   \         /                 \       /
    ●───●───●                   ●─────●

Reduces number of points while preserving shape
epsilon controls simplification level
```

---

### 10. Color Spaces

**BGR to HSV Conversion**:
```
           BGR                              HSV
    ┌───────────────┐              ┌───────────────┐
    │  Blue:0-255   │              │  Hue: 0-180   │ ◄── Color type
    │  Green:0-255  │    ──►       │  Sat: 0-255   │ ◄── Color purity
    │  Red:0-255    │              │  Val: 0-255   │ ◄── Brightness
    └───────────────┘              └───────────────┘

HSV Color Wheel:
         Red(0)
          ╱╲                        Saturation: center=0, edge=255
         ╱  ╲                       Value: dark=0, bright=255
   Yellow    Magenta
       ╲    ╱
        ╲  ╱
    Green──Cyan──Blue
        (60)  (90) (120)
```

**Common HSV Color Ranges**:
```
Color        H (Hue)        S (Sat)       V (Val)
─────────────────────────────────────────────────
Red       0-10, 170-180   100-255       100-255
Orange      10-25         100-255       100-255
Yellow      25-35         100-255       100-255
Green       35-85         100-255       100-255
Blue        85-130        100-255       100-255
Purple     130-170        100-255       100-255
```

---

### 11. Histogram Operations

#### Histogram Visualization
```
Pixel count
    ▲
200 │    ┌──┐
    │    │  │
150 │ ┌──┤  │     ┌──┐
    │ │  │  │     │  │
100 │ │  │  ├──┐  │  │
    │ │  │  │  │  │  ├──┐
 50 │ │  │  │  ├──┤  │  │
    │ │  │  │  │  │  │  │
  0 └─┴──┴──┴──┴──┴──┴──┴──►
    0  32 64 96 128 160 192 255
         Pixel intensity
```

#### Histogram Equalization
```
Before (low contrast):        After (equalized):
    ▲ Concentrated             ▲ Spread out
200 │    ████                200│ ┌┐┌┐┌┐┌┐┌┐┌┐┌┐
    │    ████                   │ ││││││││││││││
100 │    ████         ──►    100│ ││││││││││││││
    │    ████                   │ ││││││││││││││
  0 └────────────►            0 └────────────────►
    0   128   255               0   128   255

Image appears to use          Pixels spread across
only middle gray values       full intensity range
= low contrast                = better contrast
```

#### CLAHE (Adaptive Histogram Equalization)
```
Standard Equalization:        CLAHE (tile-based):
┌───────────────────┐        ┌────┬────┬────┬────┐
│                   │        │ eq │ eq │ eq │ eq │
│  Single histogram │   vs   ├────┼────┼────┼────┤
│  for entire image │        │ eq │ eq │ eq │ eq │
│                   │        ├────┼────┼────┼────┤
│                   │        │ eq │ eq │ eq │ eq │
└───────────────────┘        └────┴────┴────┴────┘
                             Each tile equalized
May over-amplify noise       separately, then
in homogeneous regions       interpolated at borders
```

---

## Filter Comparison

```
Original        Box Blur        Gaussian        Median          Bilateral
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│Sharp    │    │Uniform  │    │Smooth   │    │Salt&    │    │Smooth   │
│edges    │──► │blur     │    │blur     │    │pepper   │    │but      │
│+ noise  │    │+ edges  │    │+ edges  │    │removed  │    │edges    │
│         │    │blurred  │    │softened │    │         │    │preserved│
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
               Fastest         Good balance   Best for       Best edge
                                             impulse noise   preservation
```

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_filtering.py` | Box blur, Gaussian, median, bilateral, custom kernels, sharpening |
| `02_morphology.py` | Erosion, dilation, opening, closing, gradient, top/black hat |
| `03_edges_contours.py` | Sobel, Laplacian, Canny, contour detection and analysis |
| `04_color_histogram.py` | Color spaces, HSV, histograms, CLAHE, color segmentation |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.blur()` | Box filter |
| `cv2.GaussianBlur()` | Gaussian blur |
| `cv2.medianBlur()` | Median filter |
| `cv2.bilateralFilter()` | Edge-preserving smooth |
| `cv2.filter2D()` | Custom kernel |
| `cv2.erode()` | Morphological erosion |
| `cv2.dilate()` | Morphological dilation |
| `cv2.morphologyEx()` | Opening, closing, gradient, etc. |
| `cv2.Sobel()` | Sobel derivatives |
| `cv2.Laplacian()` | Laplacian operator |
| `cv2.Canny()` | Canny edge detection |
| `cv2.findContours()` | Find contours |
| `cv2.drawContours()` | Draw contours |
| `cv2.cvtColor()` | Color space conversion |
| `cv2.calcHist()` | Calculate histogram |
| `cv2.equalizeHist()` | Histogram equalization |
| `cv2.createCLAHE()` | Adaptive equalization |

---

## Further Reading

- [Image Processing Tutorial](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)
- [Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
