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

# Create test image
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(test_img, (200, 150), (440, 330), (0, 255, 0), -1)
cv2.circle(test_img, (320, 240), 50, (255, 0, 0), -1)

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
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display DNN module concepts."""

    # Show blob visualization
    # Reshape blob back to image for visualization
    blob_vis = blob[0].transpose(1, 2, 0)  # CHW -> HWC
    blob_vis = (blob_vis * 255).astype(np.uint8)
    blob_vis = cv2.cvtColor(blob_vis, cv2.COLOR_RGB2BGR)

    # Display
    display = np.hstack([
        cv2.resize(test_img, (224, 224)),
        blob_vis
    ])

    cv2.putText(display, "Original", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(display, "Blob", (234, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Blob Preparation", display)

    print("\nNote: This demo shows blob preparation concepts.")
    print("For actual inference, you need to download model files.")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running DNN module demonstrations...")
    print("=" * 60)
    show_demo()
