---
layout: default
title: "01: Core Functionality"
parent: Modules
nav_order: 1
permalink: /modules/01-core
---

# Module 1: Core Functionality
{: .fs-9 }

The foundation of OpenCV - understanding image representation, basic operations, and pixel manipulation.
{: .fs-6 .fw-300 }

---

## Topics Covered

- Mat structure (NumPy arrays in Python)
- Creating images programmatically
- Pixel access and manipulation
- Arithmetic operations
- Logical (bitwise) operations
- Channel splitting and merging
- Region of Interest (ROI)
- Border handling

---

## Algorithm Explanations

### 1. Image Representation (Mat/NumPy Array)

**What it does**: Represents images as multi-dimensional arrays of pixel values.

**How it works**:
- Images are stored as NumPy arrays with shape `(height, width, channels)`
- Each pixel value is typically `uint8` (0-255)
- OpenCV uses **BGR** color ordering (not RGB!)

**Image Structure Diagram**:
```
                        WIDTH (columns)
            ◄─────────────────────────────────────►
          ┌─────────────────────────────────────────┐
        ▲ │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐   │
        │ │  │BGR│ │BGR│ │BGR│ │BGR│ │BGR│ │BGR│   │
        │ │  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘   │
        │ │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐   │
 HEIGHT │ │  │BGR│ │BGR│ │BGR│ │BGR│ │BGR│ │BGR│   │
 (rows) │ │  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘   │
        │ │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐   │
        │ │  │BGR│ │BGR│ │BGR│ │BGR│ │BGR│ │BGR│   │
        ▼ │  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘   │
          └─────────────────────────────────────────┘

Each pixel: [Blue, Green, Red] = 3 bytes
```

**Coordinate System**:
```
              x (columns) ──────────────────────►
            0     1     2     3     4     5
          ┌─────┬─────┬─────┬─────┬─────┬─────┐
        0 │(0,0)│(0,1)│(0,2)│(0,3)│(0,4)│(0,5)│
    y     ├─────┼─────┼─────┼─────┼─────┼─────┤
  (rows)  1 │(1,0)│(1,1)│(1,2)│(1,3)│(1,4)│(1,5)│
    │     ├─────┼─────┼─────┼─────┼─────┼─────┤
    │     2 │(2,0)│(2,1)│(2,2)│(2,3)│(2,4)│(2,5)│
    ▼     └─────┴─────┴─────┴─────┴─────┴─────┘

    Access: image[row, col] = image[y, x]
```

**BGR vs RGB Color Ordering**:
```
    OpenCV (BGR)              Most Libraries (RGB)
    ┌───┬───┬───┐             ┌───┬───┬───┐
    │ B │ G │ R │             │ R │ G │ B │
    │[0]│[1]│[2]│             │[0]│[1]│[2]│
    └───┴───┴───┘             └───┴───┴───┘

    Example: Pure Red pixel
    OpenCV: [0, 0, 255]      RGB: [255, 0, 0]
```

---

### 2. Pixel Access

**Direct Indexing**:
```python
pixel = image[row, col]      # Returns [B, G, R] for color image
blue = image[row, col, 0]    # Blue channel only
```

**Accessing Single Pixel Components**:
```
image[2, 3] for a color image:

         col 0   col 1   col 2   col 3
       ┌───────┬───────┬───────┬───────┐
row 0  │  BGR  │  BGR  │  BGR  │  BGR  │
       ├───────┼───────┼───────┼───────┤
row 1  │  BGR  │  BGR  │  BGR  │  BGR  │
       ├───────┼───────┼───────┼───────┤
row 2  │  BGR  │  BGR  │  BGR  │ [B,G,R] ◄── image[2,3]
       └───────┴───────┴───────┴───────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              image[2,3,0]   image[2,3,1]   image[2,3,2]
                 Blue           Green          Red
```

---

### 3. Arithmetic Operations

#### Addition with Saturation

**Saturation vs Wrap-around**:
```
NumPy Addition (Wraps):           OpenCV Addition (Saturates):
    200 + 100 = 300               200 + 100 = 300
    300 % 256 = 44  ✗             min(300, 255) = 255  ✓

    ┌────────────────────┐        ┌────────────────────┐
255 │          ┌─────────│   255  │··················──┤
    │         /          │        │                 ·  │
    │        /           │        │                ·   │
    │       /            │        │               ·    │
128 │      /             │   128  │              ·     │
    │     /              │        │             ·      │
    │    /               │        │            ·       │
    │   / ← wraps to 0   │        │           ·        │
  0 │──/                 │     0  │··········          │
    └────────────────────┘        └────────────────────┘
        NumPy: modulo              OpenCV: clamp
```

#### Weighted Addition (Alpha Blending)

**Formula**: `dst = α × src1 + β × src2 + γ`

```
    Image 1 (α=0.7)         Image 2 (β=0.3)         Result
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │               │       │               │       │               │
    │    ████       │   +   │       ████    │   =   │    ▓▓▓▓▓▓     │
    │    ████       │       │       ████    │       │    ▓▓▓▓▓▓     │
    │               │       │               │       │               │
    └───────────────┘       └───────────────┘       └───────────────┘
       70% weight              30% weight            Blended
```

---

### 4. Bitwise Operations

**Bitwise Operations Overview**:
```
           AND              OR              XOR             NOT
    ┌───┬───┬───┐    ┌───┬───┬───┐    ┌───┬───┬───┐    ┌───┬───┐
    │ A │ B │OUT│    │ A │ B │OUT│    │ A │ B │OUT│    │ A │OUT│
    ├───┼───┼───┤    ├───┼───┼───┤    ├───┼───┼───┤    ├───┼───┤
    │ 0 │ 0 │ 0 │    │ 0 │ 0 │ 0 │    │ 0 │ 0 │ 0 │    │ 0 │ 1 │
    │ 0 │ 1 │ 0 │    │ 0 │ 1 │ 1 │    │ 0 │ 1 │ 1 │    │ 1 │ 0 │
    │ 1 │ 0 │ 0 │    │ 1 │ 0 │ 1 │    │ 1 │ 0 │ 1 │    └───┴───┘
    │ 1 │ 1 │ 1 │    │ 1 │ 1 │ 1 │    │ 1 │ 1 │ 0 │
    └───┴───┴───┘    └───┴───┴───┘    └───┴───┴───┘
```

#### AND Operation - Masking
```
    Image                 Mask                  Result
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│█████████████████│   │                 │   │                 │
│█████████████████│   │    ┌───────┐    │   │    ┌───────┐    │
│████ PHOTO █████│ & │    │███████│    │ = │    │ PHOTO │    │
│█████████████████│   │    │███████│    │   │    └───────┘    │
│█████████████████│   │    └───────┘    │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘
   (any content)       (white = keep)       (extracted region)
```

---

### 5. Channel Operations

#### Splitting and Merging Channels

```
                    cv2.split()
    Color Image    ─────────────►    Individual Channels
┌─────────────────┐             ┌─────────┐ ┌─────────┐ ┌─────────┐
│                 │             │  Blue   │ │  Green  │ │   Red   │
│   ┌───┐         │             │ Channel │ │ Channel │ │ Channel │
│   │BGR│ × W × H │    ───►     │  (H×W)  │ │  (H×W)  │ │  (H×W)  │
│   └───┘         │             │         │ │         │ │         │
│                 │             └─────────┘ └─────────┘ └─────────┘
└─────────────────┘                  B          G           R
    Shape: (H, W, 3)
```

---

### 6. Region of Interest (ROI)

```
              Original Image
    ┌─────────────────────────────────┐
    │                                 │
    │      (x1,y1)                    │
    │         ┌───────────────┐       │
    │         │               │       │
    │         │      ROI      │       │
    │         │               │       │
    │         └───────────────┘       │
    │                      (x2,y2)    │
    │                                 │
    └─────────────────────────────────┘

    roi = image[y1:y2, x1:x2]

    Note: Slicing is [y1:y2, x1:x2] not [x1:x2, y1:y2]!
```

---

### 7. Border Handling

```
Original: |a b c d e f g h|

BORDER_CONSTANT (value=0):
    0 0 0 0|a b c d e f g h|0 0 0 0

BORDER_REPLICATE:
    a a a a|a b c d e f g h|h h h h

BORDER_REFLECT:
    d c b a|a b c d e f g h|h g f e

BORDER_WRAP:
    e f g h|a b c d e f g h|a b c d
```

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_basics.py` | Mat creation, pixel access, arithmetic, bitwise ops, channels, ROI, borders |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
| `np.zeros((h,w,c), dtype)` | Create black image |
| `cv2.add(src1, src2)` | Saturating addition |
| `cv2.addWeighted(src1, α, src2, β, γ)` | Weighted blend |
| `cv2.bitwise_and(src1, src2)` | Bitwise AND |
| `cv2.bitwise_or(src1, src2)` | Bitwise OR |
| `cv2.split(src)` | Split channels |
| `cv2.merge([ch1, ch2, ch3])` | Merge channels |
| `cv2.copyMakeBorder(...)` | Add border |

---

## Further Reading

- [OpenCV Core Operations](https://docs.opencv.org/4.x/d7/d16/tutorial_py_table_of_contents_core.html)
- [NumPy for Image Processing](https://numpy.org/doc/stable/user/basics.html)
