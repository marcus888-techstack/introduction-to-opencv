---
layout: default
title: "05: Quality Inspection"
parent: Projects
nav_order: 5
permalink: /projects/05-quality-inspection
---

# Project 5: AI-Powered Quality Inspection System

Industrial defect detection system - detect cracks, scratches, or anomalies in products.

## What You'll Learn

1. **Image Preprocessing** - CLAHE, Gaussian blur
2. **Defect Detection** - Multiple methods
3. **Template Matching** - Comparison with reference
4. **Pass/Fail Logic** - Quality decision making
5. **Report Generation** - Documenting results

## Detection Methods

| Method | Best For |
|--------|----------|
| Threshold | Dark spots, stains |
| Edge Detection | Cracks, scratches |
| Blob Detection | Holes, bubbles |
| Comparison | Any deviation from reference |

## Usage

```bash
# Run demo with sample images
python main.py --demo

# Use camera for real-time inspection
python main.py --camera

# Inspect specific image
python main.py --image product.jpg

# With reference image
python main.py --image test.jpg --reference good.jpg
```

## Algorithm Flow

```
1. Load test image
        |
2. Preprocess
   ├── Grayscale
   ├── CLAHE enhancement
   └── Gaussian blur
        |
3. Detect defects
   ├── Adaptive threshold
   ├── Canny edges
   ├── Blob detection
   └── Reference comparison
        |
4. Classify defects
        |
5. Make decision (pass/fail)
```

## Key OpenCV Functions

| Function | Purpose |
|----------|---------|
| `cv2.createCLAHE()` | Contrast enhancement |
| `cv2.adaptiveThreshold()` | Local thresholding |
| `cv2.Canny()` | Edge detection |
| `cv2.SimpleBlobDetector()` | Blob detection |
| `cv2.absdiff()` | Image comparison |

## Real-World Applications

- Manufacturing quality control
- PCB inspection
- Food product inspection
- Textile defect detection
- Pharmaceutical packaging

## Code Highlights

### CLAHE Enhancement
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
```

### Reference Comparison
```python
diff = cv2.absdiff(reference_gray, test_gray)
_, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
```

## Quality Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_defects | 3 | Max allowed defects |
| max_area | 1000px | Max total defect area |
| critical_types | ['crack'] | Auto-fail defect types |

## References

- [OpenCV Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)
- [Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
