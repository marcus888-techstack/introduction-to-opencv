# Project 3: License Plate Recognition (ANPR)

Automatic Number Plate Recognition system for parking lots, toll booths, and security systems.

## What You'll Learn

1. **Plate Detection** - Using Haar Cascades and contour analysis
2. **Image Preprocessing** - Preparing plates for OCR
3. **OCR Integration** - Extracting text with EasyOCR
4. **Format Validation** - Verifying plate formats
5. **Logging System** - Recording vehicle entries/exits

## Key OpenCV Functions

| Function | Purpose |
|----------|---------|
| `cv2.CascadeClassifier()` | Haar cascade detection |
| `cv2.bilateralFilter()` | Noise reduction |
| `cv2.Canny()` | Edge detection |
| `cv2.findContours()` | Find plate boundaries |
| `cv2.adaptiveThreshold()` | Prepare for OCR |

## Usage

```bash
# Run demo with generated plate
python main.py --demo

# Process a single image
python main.py --image /path/to/car.jpg

# Use webcam for real-time detection
python main.py --camera

# Process video file
python main.py --video /path/to/traffic.mp4
```

## Algorithm Flow

```
1. Input Image/Video Frame
        |
2. Plate Detection
   ├── Haar Cascade
   └── Contour Analysis (fallback)
        |
3. Plate Preprocessing
   ├── Resize
   ├── Grayscale
   ├── Bilateral Filter
   └── Adaptive Threshold
        |
4. OCR (EasyOCR)
        |
5. Format Validation
        |
6. Logging & Display
```

## Detection Methods

### Haar Cascade
- Fast and lightweight
- Uses pre-trained cascade classifier
- Works well for frontal plates

### Contour Analysis
- Finds rectangular contours
- Checks aspect ratio (2.0 - 6.0)
- Works for various plate orientations

## Real-World Applications

- Parking lot management
- Toll collection systems
- Traffic monitoring
- Security checkpoints
- Law enforcement
- Access control gates

## Code Highlights

### Plate Detection
```python
# Contour-based detection
aspect_ratio = w / float(h)
if 2.0 < aspect_ratio < 6.0:  # Plate aspect ratio
    plate_candidates.append((x, y, w, h))
```

### OCR Preprocessing
```python
# Adaptive thresholding for clean OCR input
thresh = cv2.adaptiveThreshold(
    filtered, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    19, 9
)
```

## Output Format

### Log File (CSV)
```csv
PlateNumber,Confidence,Timestamp
ABC1234,0.92,2024-01-15 09:00:15
XYZ5678,0.87,2024-01-15 09:02:33
```

## Performance Tips

1. Use cascade detection first, contour as fallback
2. Process every N frames in video
3. Implement cooldown for same plate
4. Use GPU-accelerated OCR if available

## References

- [OpenCV Haar Cascades](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [Contour Analysis](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
