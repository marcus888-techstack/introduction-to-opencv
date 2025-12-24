---
layout: default
title: "10: Computational Photo"
parent: Modules
nav_order: 10
permalink: /modules/10-photo
---

# Module 10: Computational Photography
{: .fs-9 }

Image enhancement, restoration, and artistic effects using computational photography techniques.
{: .fs-6 .fw-300 }

---

## Topics Covered

- Inpainting (image restoration)
- Non-local means denoising
- HDR imaging and tone mapping
- Seamless cloning

---

## Algorithm Explanations

### 1. Inpainting

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
│   Use Cases: Remove objects, restore photos, remove watermarks    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Methods**:
- `INPAINT_NS`: Navier-Stokes (better for large regions)
- `INPAINT_TELEA`: Fast marching (faster, good for small regions)

---

### 2. Non-Local Means Denoising

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
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Similar patches exist throughout the image, not just locally.

---

### 3. HDR Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HDR Imaging Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Multiple Exposures                                                │
│   ┌────────┐ ┌────────┐ ┌────────┐                                │
│   │ Under  │ │ Normal │ │ Over   │                                │
│   │exposed │ │        │ │exposed │                                │
│   │  ████  │ │ █████  │ │██████  │                                │
│   └───┬────┘ └───┬────┘ └───┬────┘                                │
│       │          │          │                                       │
│       └──────────┴──────────┘                                       │
│                  │                                                  │
│                  ▼                                                  │
│         ┌───────────────┐                                          │
│         │  HDR Merge    │  Combine into high dynamic range         │
│         └───────┬───────┘                                          │
│                 │                                                   │
│                 ▼                                                   │
│         ┌───────────────┐                                          │
│         │  Tone Mapping │  Compress to displayable range           │
│         └───────┬───────┘                                          │
│                 │                                                   │
│                 ▼                                                   │
│         ┌───────────────┐                                          │
│         │  LDR Output   │  Ready for display                       │
│         └───────────────┘                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 4. Seamless Cloning

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Seamless Clone (Poisson Blending)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Source              Destination           Result                 │
│   ┌───────────┐       ┌───────────┐        ┌───────────┐          │
│   │   ┌───┐   │       │           │        │           │          │
│   │   │ A │   │       │   Sky     │        │   Sky     │          │
│   │   └───┘   │  +    │           │   =    │   ┌───┐   │          │
│   │ (object)  │       │  Ground   │        │   │ A │   │          │
│   └───────────┘       └───────────┘        │   └───┘   │          │
│                                             │  Ground   │          │
│                                             └───────────┘          │
│                                                                     │
│   Blending preserves gradients → no visible seams!                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_photo_basics.py` | Inpainting, denoising, HDR, seamless cloning |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
| `cv2.inpaint()` | Image inpainting |
| `cv2.fastNlMeansDenoisingColored()` | Color denoising |
| `cv2.createMergeDebevec()` | HDR merge |
| `cv2.createTonemapReinhard()` | Tone mapping |
| `cv2.seamlessClone()` | Poisson blending |

---

## Further Reading

- [Computational Photography](https://docs.opencv.org/4.x/d0/d25/tutorial_table_of_content_photo.html)
