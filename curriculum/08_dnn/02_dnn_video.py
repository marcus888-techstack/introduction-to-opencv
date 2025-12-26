"""
Module 8: Deep Learning (DNN) - Video Inference
================================================
Real-time object detection on video streams.

Official Docs: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html

Topics Covered:
1. Real-time Face Detection
2. Video Stream Processing
3. FPS Optimization
4. Multi-object Tracking
"""

import cv2
import numpy as np
import os
import sys
import urllib.request
import time

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 8: DNN Video Inference")
print("=" * 60)


# =============================================================================
# 1. MODEL SETUP - YOLOv3-tiny
# =============================================================================
print("\n--- 1. Model Setup (YOLOv3-tiny) ---")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# YOLOv3-tiny - fast and lightweight object detection
YOLO_CFG = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
YOLO_WEIGHTS = "https://pjreddie.com/media/files/yolov3-tiny.weights"
COCO_NAMES = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# COCO class labels (80 classes)
CLASSES = []


def download_model(url, filename):
    """Download model file if not exists."""
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        print(f"  Model exists: {filename}")
        return filepath
    try:
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  Downloaded: {filename}")
        return filepath
    except Exception as e:
        print(f"  Failed to download {filename}: {e}")
        return None


# Download models
cfg_path = download_model(YOLO_CFG, "yolov3-tiny.cfg")
weights_path = download_model(YOLO_WEIGHTS, "yolov3-tiny.weights")
names_path = download_model(COCO_NAMES, "coco.names")

# Load class names
if names_path and os.path.exists(names_path):
    with open(names_path, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
    print(f"  Loaded {len(CLASSES)} class names")

# Colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3)) if CLASSES else []


# =============================================================================
# 2. LOAD NETWORK
# =============================================================================
print("\n--- 2. Loading Network ---")

net = None
output_layers = []

if cfg_path and weights_path:
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    print("YOLOv3-tiny network loaded successfully")

    # Use OpenCV backend with CPU (works on all platforms including Mac)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Using OpenCV CPU backend")

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print(f"Output layers: {output_layers}")
else:
    print("Failed to load network - model files not available")


# =============================================================================
# 3. OBJECT DETECTION FUNCTION (YOLO)
# =============================================================================
print("\n--- 3. Detection Function ---")


def detect_objects(frame, confidence_threshold=0.4, nms_threshold=0.4, target_classes=None):
    """Detect objects in a frame using YOLO.

    Args:
        frame: Input image
        confidence_threshold: Minimum confidence for detection
        nms_threshold: Non-maximum suppression threshold
        target_classes: List of class names to detect (None = all classes)
    """
    if net is None or not CLASSES:
        return []

    h, w = frame.shape[:2]

    # Create blob from image (YOLO uses 1/255 scale)
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416, 416), swapRB=True, crop=False
    )

    # Run inference
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Parse YOLO output
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                class_name = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"

                # Filter by target classes if specified
                if target_classes and class_name not in target_classes:
                    continue

                # YOLO returns center x, center y, width, height
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                box_w = int(detection[2] * w)
                box_h = int(detection[3] * h)

                # Convert to corner coordinates
                x1 = int(center_x - box_w / 2)
                y1 = int(center_y - box_h / 2)

                boxes.append([x1, y1, box_w, box_h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    objects = []
    for i in indices:
        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
        box = boxes[idx]
        x1, y1, box_w, box_h = box

        objects.append({
            'box': (x1, y1, x1 + box_w, y1 + box_h),
            'confidence': confidences[idx],
            'class_id': class_ids[idx],
            'class_name': CLASSES[class_ids[idx]] if class_ids[idx] < len(CLASSES) else "unknown"
        })

    return objects


def draw_detections(frame, objects):
    """Draw detection results on frame."""
    for obj in objects:
        x1, y1, x2, y2 = obj['box']
        conf = obj['confidence']
        class_name = obj['class_name']
        class_id = obj['class_id']

        # Get color for this class
        color = [int(c) for c in COLORS[class_id]]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{class_name}: {conf:.0%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0] + 4, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# =============================================================================
# 4. VIDEO PROCESSING
# =============================================================================
print("\n--- 4. Video Processing ---")

video_info = """
Video Processing Options:

1. Webcam:
   cap = cv2.VideoCapture(0)

2. Video File:
   cap = cv2.VideoCapture('video.mp4')

3. RTSP Stream:
   cap = cv2.VideoCapture('rtsp://...')

Optimization Tips:
- Reduce frame size for faster processing
- Skip frames if needed (process every Nth frame)
- Use threading for capture and processing
- Enable GPU backend if available
"""
print(video_info)


# =============================================================================
# 5. DEMO - VIDEO FACE DETECTION
# =============================================================================
def run_video_demo():
    """Run face detection on video file."""

    if net is None:
        print("Network not loaded. Cannot run demo.")
        return

    # Load sample video
    video_path = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'vtest.avi')

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Run: python curriculum/sample_data/download_samples.py")
        return

    print(f"Using video file: vtest.avi")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {video_fps:.1f} FPS")

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    frame_num = 0

    print("\nRunning face detection on video...")
    print("Press 'q' to quit, 's' to save screenshot, SPACE to pause")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                continue

            frame_num += 1

            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))

            # Detect objects (persons and other objects)
            start_time = time.time()
            objects = detect_objects(frame, confidence_threshold=0.4)
            inference_time = (time.time() - start_time) * 1000

            # Count persons
            persons = [o for o in objects if o['class_name'] == 'person']

            # Draw results
            frame = draw_detections(frame, objects)

            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Draw info overlay
            info_text = [
                f"FPS: {fps:.1f}",
                f"Inference: {inference_time:.1f}ms",
                f"Objects: {len(objects)} (Persons: {len(persons)})",
                f"Frame: {frame_num}/{total_frames}"
            ]

            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.putText(frame, "q:quit  SPACE:pause  s:screenshot", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Display
        cv2.imshow("DNN Video Face Detection", frame)

        # Handle key press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('screenshot.png', frame)
            print("Screenshot saved!")
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo finished.")


# =============================================================================
# 6. ALTERNATIVE: IMAGE SEQUENCE DEMO
# =============================================================================
def run_image_demo():
    """Run object detection on sample images if video not available."""

    if net is None:
        print("Network not loaded. Cannot run demo.")
        return

    # Load test images
    test_images = ["fruits.jpg", "messi5.jpg", "lena.jpg"]

    for img_name in test_images:
        img = get_image(img_name)
        if img is None:
            continue

        print(f"\nProcessing: {img_name}")

        # Resize
        img = cv2.resize(img, (640, 480))

        # Detect objects
        start_time = time.time()
        objects = detect_objects(img, confidence_threshold=0.4)
        inference_time = (time.time() - start_time) * 1000

        print(f"  Inference time: {inference_time:.2f}ms")
        print(f"  Objects detected: {len(objects)}")
        for obj in objects:
            print(f"    - {obj['class_name']}: {obj['confidence']:.0%}")

        # Draw results
        result = draw_detections(img.copy(), objects)

        # Add info
        cv2.putText(result, f"{img_name} - {len(objects)} object(s)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result, f"Inference: {inference_time:.1f}ms", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(f"Object Detection - {img_name}", result)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DNN Video Inference Demo")
    print("=" * 60)

    if net is None:
        print("\nError: Model not loaded. Please check internet connection.")
        sys.exit(1)

    # Check if video exists, otherwise run image demo
    video_path = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'vtest.avi')

    if os.path.exists(video_path):
        run_video_demo()
    else:
        print("\nVideo file not found. Running image demo instead.")
        print("To get sample video, run: python curriculum/sample_data/download_samples.py")
        run_image_demo()
