---
layout: default
title: "07: QR/Barcode Reader"
parent: Applications
nav_order: 7
permalink: /applications/07-qr-barcode
---

# QR Code & Barcode Reader
{: .fs-9 }

Detect and decode QR codes and barcodes in real-time.
{: .fs-6 .fw-300 }

[View Source Code](https://github.com/marcus888-techstack/introduction-to-opencv/blob/main/curriculum/applications/07_qr_barcode_reader.py){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Overview

Scan QR codes and barcodes using OpenCV's built-in detectors. Useful for payment systems, inventory management, and ticket scanning.

**Key Techniques:**
- QRCodeDetector
- BarcodeDetector (OpenCV 4.5.3+)
- Real-time video processing

---

## How It Works

```
Frame → Detect → Decode → Display Result
   ↓       ↓        ↓          ↓
[Image]  [Find   [Read     [Show
          corners] data]    decoded text]
```

---

## Key OpenCV Functions

```python
# QR Code Detection
qr_detector = cv2.QRCodeDetector()

# Detect and decode single QR
data, points, straight_qr = qr_detector.detectAndDecode(frame)

if data:
    print(f"QR Code: {data}")
    # Draw bounding polygon
    points = points[0].astype(int)
    cv2.polylines(frame, [points], True, (0, 255, 0), 3)

# Detect multiple QR codes
retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)
```

---

## Barcode Detection (OpenCV 4.5.3+)

```python
# Create barcode detector
barcode_detector = cv2.barcode.BarcodeDetector()

# Detect and decode
retval, decoded_info, decoded_type, points = barcode_detector.detectAndDecode(frame)

if retval:
    for i, data in enumerate(decoded_info):
        print(f"Barcode ({decoded_type[i]}): {data}")
```

---

## Supported Formats

| Type | Formats |
|:-----|:--------|
| QR Code | Standard QR, Micro QR |
| 1D Barcodes | EAN-13, EAN-8, UPC-A, UPC-E |
| | Code-39, Code-93, Code-128 |
| | ITF, Codabar |

---

## Tips for Better Detection

1. **Good lighting**: Avoid shadows and glare
2. **Steady camera**: Motion blur reduces accuracy
3. **Proper distance**: Code should fill 30-70% of frame
4. **Clean codes**: Damaged codes may not scan

---

## Controls

| Key | Action |
|:----|:-------|
| `s` | Save screenshot with detection |
| `q` | Quit |

---

## Running the Application

```bash
python curriculum/applications/07_qr_barcode_reader.py
```

---

## Official Documentation

- [QRCodeDetector](https://docs.opencv.org/4.x/de/dc3/classcv_1_1QRCodeDetector.html)
- [BarcodeDetector](https://docs.opencv.org/4.x/d2/dea/group__barcode.html)
