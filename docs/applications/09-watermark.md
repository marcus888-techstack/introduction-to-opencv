---
layout: default
title: "09: Image Watermarking"
parent: Applications
nav_order: 9
permalink: /applications/09-watermark
---

# Image Watermarking
{: .fs-9 }

Add visible and invisible watermarks to protect images.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/09_image_watermark.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Add watermarks for copyright protection. Includes visible text/logo watermarks and invisible steganographic watermarks.

**Key Techniques:**
- Image blending (addWeighted)
- Alpha channel manipulation
- Bitwise operations
- LSB steganography

---

## Watermark Types

### 1. Text Watermark

```python
def text_watermark(image, text, position, opacity=0.5):
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(overlay, text, position, font, 1.0, (255, 255, 255), 2)

    # Blend original and overlay
    result = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
    return result
```

### 2. Logo Watermark

```python
def logo_watermark(image, logo, position, opacity=0.3):
    # Resize logo
    logo_resized = cv2.resize(logo, (100, 100))

    # Extract ROI
    x, y = position
    h, w = logo_resized.shape[:2]
    roi = image[y:y+h, x:x+w]

    # Blend
    blended = cv2.addWeighted(logo_resized, opacity, roi, 1 - opacity, 0)
    image[y:y+h, x:x+w] = blended

    return image
```

### 3. Tiled Watermark

```python
def tiled_watermark(image, text, spacing=150, opacity=0.1):
    overlay = np.zeros_like(image)

    for y in range(0, image.shape[0], spacing):
        for x in range(0, image.shape[1], spacing):
            cv2.putText(overlay, text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    return cv2.addWeighted(overlay, opacity, image, 1, 0)
```

---

## Invisible Watermark (LSB)

Hide data in the least significant bits of pixels:

```python
def embed_watermark(image, text):
    result = image.copy()
    binary = ''.join(format(ord(c), '08b') for c in text)
    binary += '00000000'  # Null terminator

    flat = result[:, :, 0].flatten()

    for i, bit in enumerate(binary):
        if i >= len(flat):
            break
        flat[i] = (flat[i] & 0xFE) | int(bit)  # Replace LSB

    result[:, :, 0] = flat.reshape(result[:, :, 0].shape)
    return result

def extract_watermark(image):
    flat = image[:, :, 0].flatten()
    bits = ''.join(str(flat[i] & 1) for i in range(1000))

    text = ''
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if byte == '00000000':
            break
        text += chr(int(byte, 2))

    return text
```

---

## Controls

| Key | Action |
|:----|:-------|
| `1` | Text watermark |
| `2` | Logo watermark |
| `3` | Tiled watermark |
| `4` | Invisible watermark |
| `5` | Extract invisible watermark |
| `p` | Change position |
| `+/-` | Adjust opacity |
| `s` | Save result |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/09_image_watermark.py
```

---

## Official Documentation

- [Image Arithmetics](https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html)
- [Bitwise Operations](https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html)
