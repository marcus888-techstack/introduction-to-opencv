# Introduction to OpenCV - Practical Projects Course

A hands-on OpenCV course with 6 practical, real-world projects for intermediate Python students.

## Course Structure

| Session | Project | Description |
|---------|---------|-------------|
| 1-2 | Document Scanner | Edge detection, perspective transform, OCR |
| 1-2 | Face Attendance | Face detection & recognition system |
| 3-4 | License Plate Recognition | ANPR for parking/security systems |
| 3-4 | Object Counting | People/vehicle tracking & analytics |
| 5-6 | Quality Inspection | Industrial defect detection |
| 5-6 | Gesture Control | Touchless presentation control |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a project
cd projects/01_document_scanner
python main.py
```

## Projects

### 1. Smart Document Scanner with OCR
Digitize documents with automatic edge detection and text extraction.
- Auto document boundary detection
- Perspective correction
- Text extraction to PDF

### 2. Face Recognition Attendance System
Contactless attendance system for offices/schools.
- Face registration & recognition
- Attendance logging with timestamps
- CSV/Excel export

### 3. License Plate Recognition (ANPR)
Automatic Number Plate Recognition for parking/toll systems.
- Plate localization
- Character recognition
- Vehicle database logging

### 4. Real-Time Object Counting & Tracking
People/vehicle counter for retail analytics and traffic monitoring.
- Multi-object tracking
- Direction detection (in/out)
- Heatmap visualization

### 5. AI-Powered Quality Inspection
Industrial defect detection for manufacturing.
- Defect classification
- Pass/fail decision
- Quality reports

### 6. Gesture-Controlled Presentations
Control slides using hand gestures.
- Swipe for next/previous
- Pointer simulation
- Custom gesture mapping

## Official OpenCV Documentation

- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)
- [Object Detection](https://docs.opencv.org/4.x/d2/d64/tutorial_table_of_content_objdetect.html)
- [Video Analysis](https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html)

## Requirements

- Python 3.8+
- Webcam (for real-time projects)
- See `requirements.txt` for full dependencies
