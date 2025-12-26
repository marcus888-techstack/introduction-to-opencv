---
layout: default
title: "10: Color Palette"
parent: Applications
nav_order: 10
permalink: /applications/10-color-palette
---

# Color Palette Extractor
{: .fs-9 }

Extract dominant colors from images for design and branding.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/10_color_palette_extractor.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Extract the dominant colors from any image using K-Means clustering. Useful for design tools, brand color analysis, and style matching.

**Key Techniques:**
- K-Means clustering
- Color quantization
- Color space analysis

---

## How It Works

```
Image → Reshape to Pixels → K-Means → Cluster Centers → Palette
   ↓          ↓               ↓            ↓             ↓
[HxWx3]   [N x 3]         [K clusters] [K colors]    [Swatches]
```

---

## Key OpenCV Functions

```python
# Reshape image to list of pixels
pixels = image.reshape(-1, 3).astype(np.float32)

# K-Means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(
    pixels,
    K=5,                      # Number of colors
    bestLabels=None,
    criteria=criteria,
    attempts=10,
    flags=cv2.KMEANS_RANDOM_CENTERS
)

# Get percentages
unique, counts = np.unique(labels, return_counts=True)
percentages = counts / len(labels) * 100

# Sort by dominance
sorted_idx = np.argsort(percentages)[::-1]
colors = centers[sorted_idx].astype(int)
percentages = percentages[sorted_idx]
```

---

## Create Visual Palette

```python
def create_palette(colors, percentages, width=400, height=100):
    palette = np.zeros((height, width, 3), dtype=np.uint8)

    x = 0
    for color, pct in zip(colors, percentages):
        w = int(width * pct / 100)
        palette[:, x:x+w] = color
        x += w

    return palette
```

---

## Color Swatches with Hex Codes

```python
def create_swatches(colors, percentages, swatch_size=80):
    n = len(colors)
    padding = 10
    width = n * (swatch_size + padding) + padding
    height = swatch_size + 60

    swatches = np.ones((height, width, 3), dtype=np.uint8) * 255

    for i, (color, pct) in enumerate(zip(colors, percentages)):
        x = padding + i * (swatch_size + padding)

        # Draw swatch
        cv2.rectangle(swatches, (x, padding),
                     (x + swatch_size, padding + swatch_size),
                     color.tolist(), -1)

        # Add hex code
        hex_code = '#{:02X}{:02X}{:02X}'.format(color[2], color[1], color[0])
        cv2.putText(swatches, hex_code, (x, height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return swatches
```

---

## Color Quantization

Reduce image to palette colors:

```python
def quantize_image(image, n_colors):
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)

    quantized = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
    return quantized
```

---

## Controls

| Key | Action |
|:----|:-------|
| `3-9` | Set number of colors |
| `q` | Quantize image |
| `c` | Show complementary colors |
| `s` | Save palette |
| `r` | Reset |
| `ESC` | Quit |

---

## Running the Application

```bash
python curriculum/applications/10_color_palette_extractor.py
```

---

## Official Documentation

- [K-Means Clustering](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html)
