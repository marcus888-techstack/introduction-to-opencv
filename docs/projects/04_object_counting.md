---
layout: default
title: "04: Object Counting"
parent: Projects
nav_order: 4
---

# Project 4: Real-Time Object Counting & Tracking

Build a people/vehicle counter for retail analytics, traffic monitoring, or warehouse management.

## What You'll Learn

1. **Object Detection** - Using YOLO for real-time detection
2. **Multi-Object Tracking** - Centroid-based tracking algorithm
3. **Counting Logic** - Line crossing detection
4. **Direction Detection** - In/out movement tracking
5. **Analytics** - Generating traffic statistics

## Key Concepts

| Concept | Description |
|---------|-------------|
| Detection | Finding objects in each frame |
| Tracking | Maintaining object identity across frames |
| Counting Line | Virtual line for crossing detection |
| Centroid | Center point of bounding box |

## Usage

```bash
# Run demo with simulated objects
python main.py --demo

# Use webcam
python main.py --camera

# Process video file
python main.py --video traffic.mp4

# Count specific classes
python main.py --camera --classes person car
```

## Algorithm

```
1. Detect objects in frame (YOLO)
        |
2. Extract centroids
        |
3. Match with existing tracks
   (Hungarian algorithm / nearest neighbor)
        |
4. Update track positions
        |
5. Check line crossing
        |
6. Update counts
```

## Tracking Algorithm

Simple centroid tracker:
1. Get centroids from new detections
2. Compare with existing object positions
3. Match based on minimum distance
4. Handle new objects and disappeared objects

## Real-World Applications

- Retail foot traffic analysis
- Traffic monitoring
- Warehouse inventory tracking
- Crowd management
- Parking lot occupancy

## Code Highlights

### Centroid Matching
```python
# Compute distances between existing and new centroids
for i, obj_c in enumerate(object_centroids):
    for j, inp_c in enumerate(input_centroids):
        D[i, j] = np.sqrt((obj_c[0] - inp_c[0])**2 +
                          (obj_c[1] - inp_c[1])**2)
```

### Line Crossing Detection
```python
# Check if object crossed the line
if prev[1] < line_y <= centroid[1]:
    count_down += 1  # Crossed downward
elif prev[1] > line_y >= centroid[1]:
    count_up += 1    # Crossed upward
```

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Video Analysis](https://docs.opencv.org/4.x/da/dd0/tutorial_table_of_content_video.html)
