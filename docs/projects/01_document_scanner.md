---
layout: default
title: "01: Document Scanner"
parent: Projects
nav_order: 1
---

# Project 1: Smart Document Scanner with OCR

A mobile-scanner-like app that digitizes documents with automatic edge detection and text extraction.

## What You'll Learn

1. **Edge Detection** - Using Canny algorithm to find document edges
2. **Contour Detection** - Finding and filtering contours to locate documents
3. **Perspective Transformation** - Converting tilted documents to flat, top-down view
4. **Image Enhancement** - Adaptive thresholding for clean output
5. **OCR Integration** - Extracting text from scanned documents

## Key OpenCV Functions

| Function | Purpose |
|----------|---------|
| `cv2.Canny()` | Detect edges in image |
| `cv2.findContours()` | Find contours from edges |
| `cv2.approxPolyDP()` | Approximate contour to polygon |
| `cv2.getPerspectiveTransform()` | Compute transformation matrix |
| `cv2.warpPerspective()` | Apply perspective transform |
| `cv2.adaptiveThreshold()` | Enhance document for reading |

## Usage

```bash
# Run demo with generated sample image
python main.py --demo

# Scan a specific image
python main.py --image /path/to/document.jpg

# Use webcam for real-time scanning
python main.py --camera

# Show debug visualization
python main.py --demo --debug
```

## Algorithm Steps

```
1. Load Image
     |
2. Convert to Grayscale
     |
3. Apply Gaussian Blur
     |
4. Canny Edge Detection
     |
5. Find Contours
     |
6. Filter for 4-point contours
     |
7. Order corner points
     |
8. Apply Perspective Transform
     |
9. Enhance with Adaptive Threshold
     |
10. Extract Text (OCR)
```

## Real-World Applications

- Mobile scanning apps (CamScanner, Adobe Scan)
- Document digitization systems
- Receipt scanning for expense tracking
- ID/passport scanning
- Whiteboard capture

## Code Highlights

### Finding Document Corners
```python
# Approximate contour to polygon
peri = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

# If 4 points, it's likely a document
if len(approx) == 4:
    doc_contour = approx
```

### Perspective Transform
```python
# Define destination points (top-down view)
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]])

# Get transformation matrix and apply
M = cv2.getPerspectiveTransform(src_pts, dst)
warped = cv2.warpPerspective(image, M, (width, height))
```

## References

- [OpenCV Contour Tutorial](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
