---
layout: default
title: "02: Face Attendance"
parent: Projects
nav_order: 2
---

# Project 2: Face Recognition Attendance System

A contactless attendance system using face recognition - widely used in offices and schools.

## What You'll Learn

1. **Face Detection** - Locating faces in images using Haar Cascades or DNN
2. **Face Encoding** - Converting faces to numerical feature vectors
3. **Face Recognition** - Comparing face encodings to identify people
4. **Database Management** - Storing and retrieving face data
5. **Attendance Logging** - Recording attendance with timestamps

## Key Concepts

| Concept | Description |
|---------|-------------|
| Face Detection | Finding faces in an image |
| Face Encoding | 128-dimensional vector representing a face |
| Face Matching | Comparing encodings using distance metrics |
| Tolerance | Threshold for considering faces as match |

## Usage

```bash
# Register new faces
python main.py --register

# Start attendance mode
python main.py --attendance

# List registered faces
python main.py --list

# Export attendance to Excel
python main.py --export

# Run demo mode
python main.py --demo
```

## System Flow

```
Registration Mode:
1. Capture face from camera
2. Detect face location
3. Extract face encoding (128-d vector)
4. Store encoding with name in database

Attendance Mode:
1. Capture frame from camera
2. Detect all faces
3. Extract encodings
4. Compare with registered encodings
5. Identify matches and mark attendance
6. Log to CSV file
```

## Key OpenCV/ML Functions

| Function | Library | Purpose |
|----------|---------|---------|
| `face_recognition.face_locations()` | face_recognition | Detect face locations |
| `face_recognition.face_encodings()` | face_recognition | Get face embeddings |
| `face_recognition.face_distance()` | face_recognition | Compare face embeddings |
| `cv2.CascadeClassifier()` | OpenCV | Haar cascade detection |
| `pickle.dump()` | Python | Save database to file |

## Real-World Applications

- Office attendance systems
- School/university attendance
- Event check-in systems
- Access control systems
- Visitor management

## Code Highlights

### Face Encoding
```python
# Detect faces
face_locations = face_recognition.face_locations(rgb_image)

# Get 128-dimensional encoding
encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
```

### Face Matching
```python
# Compare with known faces
distances = face_recognition.face_distance(known_encodings, test_encoding)

# Lower distance = better match
if distances[best_match_idx] < tolerance:
    name = known_names[best_match_idx]
```

## Database Structure

```
face_database/
├── encodings.pkl      # Pickled face encodings + names
├── Alice.jpg          # Reference photo
├── Bob.jpg
└── Charlie.jpg

attendance_logs/
├── attendance_2024-01-15.csv
├── attendance_2024-01-16.csv
└── ...
```

## Output Format

### CSV Log
```csv
Name,Timestamp
Alice,2024-01-15 09:00:15
Bob,2024-01-15 09:02:33
Charlie,2024-01-15 09:05:12
```

## Performance Tips

1. Process every N frames (not every frame)
2. Resize images before processing
3. Use `model="cnn"` for GPU acceleration
4. Pre-compute encodings during registration

## References

- [face_recognition library](https://github.com/ageitgey/face_recognition)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/d2/d64/tutorial_table_of_content_objdetect.html)
- [dlib Face Recognition](http://dlib.net/face_recognition.py.html)
