"""
Module 8: Deep Learning (DNN) - Basics
======================================
Using deep neural networks in OpenCV.

Official Docs: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html

Topics Covered:
1. DNN Module Overview
2. Loading Models
3. Blob Preparation
4. Inference
5. Common Architectures
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 8: Deep Learning with OpenCV DNN")
print("=" * 60)


# =============================================================================
# 1. DNN MODULE OVERVIEW
# =============================================================================
print("\n--- 1. DNN Module Overview ---")

overview = """
OpenCV DNN Module:

Supported Frameworks:
  - TensorFlow (.pb, .pbtxt)
  - Caffe (.prototxt, .caffemodel)
  - Darknet/YOLO (.cfg, .weights)
  - ONNX (.onnx)
  - PyTorch (via ONNX export)

Supported Layers:
  - Convolution, Pooling, BatchNorm
  - Fully Connected, Softmax
  - LSTM, GRU (limited)
  - Attention (newer versions)

Backends:
  - OpenCV (default, CPU)
  - CUDA (GPU acceleration)
  - OpenVINO (Intel optimization)

Loading Functions:
  cv2.dnn.readNet()           - Auto-detect format
  cv2.dnn.readNetFromCaffe()  - Caffe models
  cv2.dnn.readNetFromTensorflow() - TensorFlow models
  cv2.dnn.readNetFromDarknet()    - YOLO models
  cv2.dnn.readNetFromONNX()   - ONNX models
"""
print(overview)


# =============================================================================
# 2. BLOB PREPARATION
# =============================================================================
print("\n--- 2. Blob Preparation ---")


def load_dnn_test_image():
    """Load a real image for DNN demo or create fallback."""
    # Try to load real images - prefer images with faces for face detection demo
    # OpenCV samples:
    # - messi5.jpg: person with face (for face detection)
    # - lena.jpg: classic face image
    # - fruits.jpg: food items (for classification)
    for sample in ["messi5.jpg", "lena.jpg", "fruits.jpg", "baboon.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return cv2.resize(img, (640, 480))

    # Fallback: Create test image
    print("No sample image found. Using synthetic image.")
    print("Run: python curriculum/sample_data/download_samples.py")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (200, 150), (440, 330), (0, 255, 0), -1)
    cv2.circle(img, (320, 240), 50, (255, 0, 0), -1)
    return img


test_img = load_dnn_test_image()

# Convert image to blob
# blobFromImage parameters:
#   image       - Input image
#   scalefactor - Scale factor (1/255 for normalization)
#   size        - Output size (width, height)
#   mean        - Mean subtraction values (BGR)
#   swapRB      - Swap Red and Blue channels
#   crop        - Whether to crop

blob = cv2.dnn.blobFromImage(
    test_img,
    scalefactor=1/255.0,
    size=(224, 224),
    mean=(0, 0, 0),
    swapRB=True,
    crop=False
)

print(f"Original image shape: {test_img.shape}")
print(f"Blob shape: {blob.shape}")  # (1, 3, 224, 224) - NCHW format

blob_info = """
Blob Format (NCHW):
  N - Batch size
  C - Channels (3 for RGB)
  H - Height
  W - Width

Common Preprocessing:
  ImageNet: mean=(123.68, 116.779, 103.939), scalefactor=1.0
  SSD: mean=(104.0, 177.0, 123.0), scalefactor=1.0
  YOLO: mean=(0,0,0), scalefactor=1/255.0

Note: swapRB=True converts BGR (OpenCV) to RGB (most DNNs)
"""
print(blob_info)


# =============================================================================
# 3. MODEL LOADING (Conceptual)
# =============================================================================
print("\n--- 3. Loading Models ---")

loading_info = """
Loading Different Model Formats:

# Caffe
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',    # Architecture
    'model.caffemodel'    # Weights
)

# TensorFlow
net = cv2.dnn.readNetFromTensorflow(
    'frozen_graph.pb',    # Frozen model
    'graph.pbtxt'         # Optional: text graph
)

# Darknet (YOLO)
net = cv2.dnn.readNetFromDarknet(
    'yolov3.cfg',         # Configuration
    'yolov3.weights'      # Weights
)

# ONNX
net = cv2.dnn.readNetFromONNX('model.onnx')

# Auto-detect
net = cv2.dnn.readNet('model_file', 'config_file')
"""
print(loading_info)


# =============================================================================
# 4. INFERENCE PROCESS
# =============================================================================
print("\n--- 4. Inference Process ---")

inference_steps = """
DNN Inference Steps:

1. Load the model:
   net = cv2.dnn.readNet(model_path)

2. Set backend/target (optional):
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

3. Prepare input blob:
   blob = cv2.dnn.blobFromImage(image, scale, size, mean)

4. Set input:
   net.setInput(blob)

5. Forward pass:
   output = net.forward()  # Single output layer
   outputs = net.forward(outputLayers)  # Multiple layers

6. Process output:
   - Classification: argmax for class
   - Detection: parse bounding boxes
   - Segmentation: process masks
"""
print(inference_steps)


# =============================================================================
# 5. COMMON ARCHITECTURES
# =============================================================================
print("\n--- 5. Common DNN Architectures ---")

architectures = """
Image Classification:
  - MobileNet: Fast, mobile-friendly
  - ResNet: Very accurate, deeper networks
  - VGG: Classic, well-understood
  - EfficientNet: Best accuracy/speed tradeoff

Object Detection:
  - YOLO (v3-v8): Real-time, good accuracy
  - SSD: Fast single-shot detector
  - Faster R-CNN: High accuracy, slower
  - RetinaNet: Good for small objects

Semantic Segmentation:
  - FCN: Fully convolutional
  - U-Net: Great for medical images
  - DeepLab: State-of-the-art

Face Detection:
  - SSD with ResNet10
  - MTCNN
  - RetinaFace

Pose Estimation:
  - OpenPose
  - MediaPipe Pose (not in cv2.dnn)
"""
print(architectures)


# =============================================================================
# 6. EXAMPLE: Simple Classification (Conceptual)
# =============================================================================
print("\n--- 6. Classification Example (Conceptual) ---")

classification_example = '''
# Example: Image Classification with MobileNet

# Load model
net = cv2.dnn.readNetFromCaffe(
    'mobilenet_deploy.prototxt',
    'mobilenet.caffemodel'
)

# Prepare image
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=0.007843,
    size=(224, 224),
    mean=(127.5, 127.5, 127.5)
)

# Inference
net.setInput(blob)
predictions = net.forward()

# Get top prediction
class_id = np.argmax(predictions[0])
confidence = predictions[0][class_id]

print(f"Class: {class_id}, Confidence: {confidence:.2%}")
'''
print(classification_example)


# =============================================================================
# 7. EXAMPLE: Object Detection (Conceptual)
# =============================================================================
print("\n--- 7. Object Detection Example (Conceptual) ---")

detection_example = '''
# Example: YOLO Object Detection

# Load model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Prepare image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True)

# Inference
net.setInput(blob)
outputs = net.forward(output_layers)

# Process detections
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # Get bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Draw box
            x = center_x - w // 2
            y = center_y - h // 2
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
'''
print(detection_example)


# =============================================================================
# 8. PERFORMANCE TIPS
# =============================================================================
print("\n--- 8. Performance Tips ---")

performance_tips = """
Performance Optimization:

1. Use GPU backend (if available):
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

2. Use OpenVINO (Intel CPUs):
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

3. Reduce input size:
   - Smaller blobs = faster inference
   - Trade-off with accuracy

4. Batch processing:
   - Process multiple images at once
   - Better GPU utilization

5. Model optimization:
   - Quantization (INT8)
   - Pruning
   - Knowledge distillation

Profiling:
  timing = net.getPerfProfile()  # Get inference time
  print(f"Inference time: {timing[0] * 1000 / cv2.getTickFrequency():.2f} ms")
"""
print(performance_tips)


# =============================================================================
# 9. DOWNLOAD MODEL FILES (YOLOv3-tiny)
# =============================================================================
print("\n--- 9. Model Download ---")

import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# YOLOv3-tiny - fast and lightweight
YOLO_CFG = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
YOLO_WEIGHTS = "https://pjreddie.com/media/files/yolov3-tiny.weights"
COCO_NAMES = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"


def download_model(url, filename):
    """Download model file if not exists."""
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        return filepath
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  Downloaded: {filename}")
        return filepath
    except Exception as e:
        print(f"  Failed to download {filename}: {e}")
        return None


# =============================================================================
# VISUALIZATION WITH ACTUAL INFERENCE (YOLO)
# =============================================================================
def show_demo():
    """Display DNN module with YOLO object detection."""

    result_img = test_img.copy()

    # Download YOLO model
    cfg_path = download_model(YOLO_CFG, "yolov3-tiny.cfg")
    weights_path = download_model(YOLO_WEIGHTS, "yolov3-tiny.weights")
    names_path = download_model(COCO_NAMES, "coco.names")

    if cfg_path and weights_path and names_path:
        print("\nRunning YOLO Object Detection...")

        # Load class names
        with open(names_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # Load network
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get output layers
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Prepare input blob
        h, w = test_img.shape[:2]
        input_blob = cv2.dnn.blobFromImage(
            test_img, 1/255.0, (416, 416), swapRB=True, crop=False
        )

        # Run inference
        net.setInput(input_blob)
        outputs = net.forward(output_layers)

        # Get inference time
        t, _ = net.getPerfProfile()
        inference_time = t * 1000.0 / cv2.getTickFrequency()
        print(f"  Inference time: {inference_time:.2f} ms")

        # Process detections
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    box_w = int(detection[2] * w)
                    box_h = int(detection[3] * h)
                    x1 = int(center_x - box_w / 2)
                    y1 = int(center_y - box_h / 2)
                    boxes.append([x1, y1, box_w, box_h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

        # Draw detections
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        num_objects = 0
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            x1, y1, bw, bh = boxes[idx]
            label = f"{classes[class_ids[idx]]}: {confidences[idx]:.0%}"
            color = [int(c) for c in colors[class_ids[idx]]]
            cv2.rectangle(result_img, (x1, y1), (x1+bw, y1+bh), color, 2)
            cv2.putText(result_img, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            num_objects += 1

        print(f"  Objects detected: {num_objects}")

        # Add info overlay
        cv2.putText(result_img, f"YOLOv3-tiny Object Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_img, f"Inference: {inference_time:.1f}ms", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(result_img, f"Objects: {num_objects}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        print("\nModel files not available. Showing input only.")
        cv2.putText(result_img, "Model not loaded", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display results
    cv2.imshow("DNN Inference Result", result_img)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running DNN module demonstrations...")
    print("=" * 60)
    show_demo()
