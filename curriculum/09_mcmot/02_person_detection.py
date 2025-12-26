"""
Module 10: MCMOT - Person Detection
====================================
Detecting persons using YOLO with OpenCV DNN.

Official Docs: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html

Topics Covered:
1. YOLO Architecture Overview
2. Loading YOLO with cv2.dnn
3. Detection Pipeline
4. Non-Maximum Suppression (NMS)
5. Cropping Person Images for Re-ID
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_sample_path

print("=" * 60)
print("Module 10: Person Detection with YOLO")
print("=" * 60)


# =============================================================================
# 1. YOLO ARCHITECTURE OVERVIEW
# =============================================================================
print("\n--- 1. YOLO Architecture Overview ---")

yolo_info = """
YOLO (You Only Look Once):

Key Concept: Single-shot detector - processes entire image in one pass

Architecture (Simplified):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Input Image (416x416)                                                   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │  Backbone   │  Feature extraction (Darknet/CSPDarknet)               │
│  │   (CNN)     │                                                        │
│  └─────┬───────┘                                                        │
│        │                                                                 │
│        ▼                                                                 │
│  ┌─────────────┐                                                        │
│  │    Neck     │  Feature Pyramid Network (FPN)                         │
│  │  (FPN/PAN)  │  Multi-scale feature fusion                            │
│  └─────┬───────┘                                                        │
│        │                                                                 │
│        ▼                                                                 │
│  ┌─────────────┐                                                        │
│  │    Head     │  Detection heads at multiple scales                    │
│  │ (Detection) │  Output: [x, y, w, h, confidence, class_probs...]     │
│  └─────────────┘                                                        │
│                                                                          │
│  Output Layers: 3 scales (13x13, 26x26, 52x52)                          │
│  - 13x13: Large objects                                                 │
│  - 26x26: Medium objects                                                │
│  - 52x52: Small objects                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

YOLOv4-tiny (What we use):
  - Lightweight version for real-time
  - Input: 416x416 RGB
  - ~23MB weights
  - ~40 FPS on CPU (varies by hardware)
  - 80 COCO classes (class 0 = "person")
"""
print(yolo_info)


# =============================================================================
# 2. LOADING YOLO WITH CV2.DNN
# =============================================================================
print("\n--- 2. Loading YOLO with cv2.dnn ---")


def load_yolo_model():
    """
    Load YOLOv4-tiny model with OpenCV DNN.

    Returns:
        net: OpenCV DNN network
        output_layers: Names of output layers
        classes: List of class names
    """
    # Get model file paths
    weights_path = get_sample_path("yolov4-tiny.weights")
    cfg_path = get_sample_path("yolov4-tiny.cfg")
    names_path = get_sample_path("coco.names")

    # Check if files exist
    files_exist = all([
        os.path.exists(weights_path),
        os.path.exists(cfg_path),
        os.path.exists(names_path)
    ])

    if not files_exist:
        print("  Model files not found! Run download_samples.py first.")
        print("  Required files:")
        print(f"    - {weights_path}")
        print(f"    - {cfg_path}")
        print(f"    - {names_path}")
        return None, None, None

    # Load network
    print("  Loading YOLO network...")
    net = cv2.dnn.readNet(weights_path, cfg_path)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load class names
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    print(f"  Loaded YOLOv4-tiny: {len(classes)} classes")
    print(f"  Output layers: {output_layers}")

    return net, output_layers, classes


# Try to load model
net, output_layers, classes = load_yolo_model()

if net is not None:
    # Set backend (use CUDA if available)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # CPU fallback
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("  Using CPU backend")


# =============================================================================
# 3. DETECTION PIPELINE
# =============================================================================
print("\n--- 3. Detection Pipeline ---")

detection_info = """
YOLO Detection Pipeline:

1. PREPROCESS (blobFromImage)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Original Image                    Blob                            │
   │  ┌─────────────┐                  ┌─────────────┐                  │
   │  │ Any size    │  ──────────────> │ 416x416x3   │                  │
   │  │ BGR         │   - Resize       │ Normalized  │                  │
   │  │ [0-255]     │   - Swap RB      │ [0-1]       │                  │
   │  └─────────────┘   - Normalize    └─────────────┘                  │
   └─────────────────────────────────────────────────────────────────────┘

2. FORWARD PASS
   blob → net.setInput() → net.forward() → outputs

3. PARSE OUTPUTS
   Each detection: [center_x, center_y, width, height, obj_conf, class1_conf, ...]
   - Coordinates are normalized (0-1)
   - Filter by confidence threshold
   - Get class with highest probability

4. NON-MAXIMUM SUPPRESSION (NMS)
   Remove overlapping boxes, keep only best detections
"""
print(detection_info)


def detect_objects(net, output_layers, image, conf_threshold=0.5, nms_threshold=0.4):
    """
    Run YOLO detection on an image.

    Args:
        net: OpenCV DNN network
        output_layers: Output layer names
        image: Input image (BGR)
        conf_threshold: Minimum confidence threshold
        nms_threshold: NMS threshold

    Returns:
        List of detections: [(class_id, confidence, (x, y, w, h)), ...]
    """
    height, width = image.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1/255.0,
        size=(416, 416),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    # Forward pass
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Parse outputs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                # Scale bounding box to original image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Top-left corner
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Build result list
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                'class_id': class_ids[i],
                'confidence': confidences[i],
                'box': tuple(boxes[i])  # (x, y, w, h)
            })

    return detections


# =============================================================================
# 4. NON-MAXIMUM SUPPRESSION (NMS)
# =============================================================================
print("\n--- 4. Non-Maximum Suppression (NMS) ---")

nms_info = """
NMS - Remove Redundant Detections:

Problem: YOLO often produces multiple overlapping boxes for same object

Before NMS:                     After NMS:
┌─────────────────────┐         ┌─────────────────────┐
│ ┌───┐               │         │ ┌───┐               │
│ │┌──┴─┐             │         │ │   │               │
│ ││    │  Person     │  ─────> │ │   │  Person       │
│ │└──┬─┘             │         │ │   │               │
│ └───┘               │         │ └───┘               │
│ 3 overlapping boxes │         │ 1 best box          │
└─────────────────────┘         └─────────────────────┘

NMS Algorithm:
1. Sort detections by confidence (highest first)
2. Keep highest confidence detection
3. Remove all detections with IoU > threshold with kept detection
4. Repeat for remaining detections

cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
  - boxes: List of [x, y, w, h]
  - confidences: List of confidence scores
  - conf_threshold: Minimum confidence to keep
  - nms_threshold: IoU threshold for suppression (typically 0.4-0.5)
"""
print(nms_info)


# =============================================================================
# 5. PERSON DETECTION (FILTER CLASS 0)
# =============================================================================
print("\n--- 5. Person Detection ---")


def detect_persons(net, output_layers, image, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detect only persons in an image.

    Args:
        net: OpenCV DNN network
        output_layers: Output layer names
        image: Input image (BGR)
        conf_threshold: Minimum confidence threshold
        nms_threshold: NMS threshold

    Returns:
        List of person detections: [(confidence, (x, y, w, h)), ...]
    """
    # Get all detections
    all_detections = detect_objects(net, output_layers, image, conf_threshold, nms_threshold)

    # Filter for person class (class_id = 0 in COCO)
    person_detections = [
        det for det in all_detections
        if det['class_id'] == 0  # person
    ]

    return person_detections


def draw_person_detections(image, detections, classes=None):
    """
    Draw person detections on image.

    Args:
        image: Input image
        detections: List of detection dicts
        classes: Optional class names list

    Returns:
        Annotated image
    """
    result = image.copy()

    for i, det in enumerate(detections):
        x, y, w, h = det['box']
        conf = det['confidence']
        class_id = det['class_id']

        # Color based on class (green for person)
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)

        # Draw box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Label
        class_name = classes[class_id] if classes else f"Class {class_id}"
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(result, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result


# =============================================================================
# 6. CROPPING PERSONS FOR RE-ID
# =============================================================================
print("\n--- 6. Cropping Persons for Re-ID ---")

crop_info = """
Preparing Person Crops for Re-ID:

Detection Output:          Crop for Re-ID:
┌─────────────────────┐    ┌─────────────┐
│                     │    │             │
│    ┌───────┐        │    │             │
│    │       │        │    │   Person    │
│    │Person │   ───> │    │   Crop      │
│    │       │        │    │             │
│    └───────┘        │    │             │
│                     │    │  128 x 256  │
└─────────────────────┘    └─────────────┘

Re-ID Model Input:
  - Size: 128 x 256 (width x height)
  - Format: RGB normalized
  - Aspect ratio: ~1:2 (person standing)

Important:
  - Add padding if aspect ratio is wrong
  - Handle edge cases (box outside image)
  - Maintain quality (don't upscale too much)
"""
print(crop_info)


def crop_person(image, box, target_size=(128, 256)):
    """
    Crop person from image for Re-ID.

    Args:
        image: Full image
        box: (x, y, w, h) bounding box
        target_size: (width, height) for output

    Returns:
        Cropped and resized person image
    """
    x, y, w, h = box
    img_h, img_w = image.shape[:2]

    # Clamp to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    # Check if valid crop
    if x2 <= x1 or y2 <= y1:
        return None

    # Crop
    crop = image[y1:y2, x1:x2]

    # Resize to target size (maintain aspect ratio with padding)
    crop_h, crop_w = crop.shape[:2]
    target_w, target_h = target_size

    # Calculate scale to fit
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    # Resize
    resized = cv2.resize(crop, (new_w, new_h))

    # Create padded output
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result


def extract_person_crops(image, detections, target_size=(128, 256)):
    """
    Extract all person crops from image.

    Args:
        image: Full image
        detections: List of detection dicts
        target_size: Output size for crops

    Returns:
        List of (crop, detection) tuples
    """
    crops = []

    for det in detections:
        if det['class_id'] != 0:  # Only persons
            continue

        crop = crop_person(image, det['box'], target_size)
        if crop is not None:
            crops.append((crop, det))

    return crops


# =============================================================================
# 7. DEMO WITH SYNTHETIC OR REAL VIDEO
# =============================================================================
print("\n--- 7. Detection Demo ---")


def create_test_image_with_persons():
    """Create a test image with simulated person-like rectangles."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background

    # Add some "persons" (colored rectangles)
    # Person 1
    cv2.rectangle(img, (100, 100), (160, 280), (50, 100, 150), -1)
    cv2.circle(img, (130, 120), 20, (200, 180, 160), -1)  # Head

    # Person 2
    cv2.rectangle(img, (300, 150), (380, 350), (80, 60, 120), -1)
    cv2.circle(img, (340, 170), 25, (200, 180, 160), -1)  # Head

    # Person 3 (smaller, in background)
    cv2.rectangle(img, (500, 200), (540, 300), (70, 90, 100), -1)
    cv2.circle(img, (520, 210), 15, (200, 180, 160), -1)  # Head

    return img


def run_detection_demo():
    """Run person detection demo."""
    print("\n  Running detection demo...")

    if net is None:
        print("  Model not loaded. Using synthetic visualization.")
        # Show synthetic demonstration
        demo_img = create_test_image_with_persons()
        cv2.putText(demo_img, "Model not loaded - synthetic demo", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(demo_img, "Run: python download_samples.py", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return [demo_img]

    # Try to use sample video
    video_path = get_sample_path("vtest.avi")

    if os.path.exists(video_path):
        print(f"  Using video: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("  Video not found. Using synthetic demo.")
        return [create_test_image_with_persons()]

    result_frames = []
    frame_count = 0
    max_frames = 50  # Limit for demo

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run person detection
        detections = detect_persons(net, output_layers, frame)

        # Draw detections
        vis_frame = draw_person_detections(frame, detections, classes)

        # Add info
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"Persons: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        result_frames.append(vis_frame)

        # Extract crops (for demonstration)
        crops = extract_person_crops(frame, detections)
        if crops and frame_count == 1:
            print(f"  Extracted {len(crops)} person crops from first frame")

    cap.release()
    print(f"  Processed {frame_count} frames")

    return result_frames


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display person detection demonstrations."""
    print("\n" + "=" * 60)
    print("Running Person Detection Demo...")
    print("=" * 60)

    result_frames = run_detection_demo()

    if not result_frames:
        print("No frames to display.")
        return

    print("\nPress any key to step through frames, ESC to exit...")

    for frame in result_frames:
        cv2.imshow("Person Detection", frame)
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running person detection demonstrations...")
    print("=" * 60)
    show_demo()
