"""
Module 8: Deep Learning (DNN) - Model Formats
==============================================
Overview of different model formats supported by OpenCV DNN.

Official Docs: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html

Topics Covered:
1. Darknet (YOLO)
2. ONNX
3. TensorFlow
4. Caffe
5. Model Conversion
"""

import cv2
import numpy as np
import os

print("=" * 60)
print("Module 8: DNN Model Formats")
print("=" * 60)


# =============================================================================
# 1. DARKNET (YOLO) FORMAT
# =============================================================================
print("\n--- 1. Darknet Format (YOLO) ---")

darknet_info = """
Darknet Format (.cfg + .weights):

Files:
  - .cfg    : Network architecture (text)
  - .weights: Trained weights (binary)

Loading:
  net = cv2.dnn.readNetFromDarknet('yolo.cfg', 'yolo.weights')

Example (YOLOv3-tiny):
  cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
  weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"

  net = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

  # Get output layers
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

  # Prepare blob (YOLO uses 1/255 scale, 416x416 input)
  blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)

  # Inference
  net.setInput(blob)
  outputs = net.forward(output_layers)

Popular YOLO Models:
  - YOLOv3-tiny: Fast, ~35MB weights
  - YOLOv3: Accurate, ~240MB weights
  - YOLOv4: Better accuracy
  - YOLOv4-tiny: Fast alternative
"""
print(darknet_info)


# =============================================================================
# 2. ONNX FORMAT
# =============================================================================
print("\n--- 2. ONNX Format ---")

onnx_info = """
ONNX Format (.onnx):

Open Neural Network Exchange - universal format.

Loading:
  net = cv2.dnn.readNetFromONNX('model.onnx')

Converting YOLO to ONNX:
  # Using ultralytics (YOLOv5/v8)
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')
  model.export(format='onnx')

  # Then load in OpenCV
  net = cv2.dnn.readNetFromONNX('yolov8n.onnx')

Converting PyTorch to ONNX:
  import torch
  model = ...  # Your PyTorch model
  dummy_input = torch.randn(1, 3, 416, 416)
  torch.onnx.export(model, dummy_input, 'model.onnx',
                    input_names=['input'],
                    output_names=['output'])

ONNX Model Sources:
  - ONNX Model Zoo: github.com/onnx/models
  - Hugging Face: huggingface.co
  - Ultralytics: github.com/ultralytics/ultralytics
"""
print(onnx_info)


# =============================================================================
# 3. TENSORFLOW FORMAT
# =============================================================================
print("\n--- 3. TensorFlow Format ---")

tf_info = """
TensorFlow Format (.pb / .pbtxt):

Files:
  - .pb    : Frozen graph (binary, weights embedded)
  - .pbtxt : Graph definition (text, optional)

Loading:
  # Single file
  net = cv2.dnn.readNetFromTensorflow('model.pb')

  # With config
  net = cv2.dnn.readNetFromTensorflow('model.pb', 'config.pbtxt')

Converting TensorFlow to ONNX (then use in OpenCV):
  pip install tf2onnx

  python -m tf2onnx.convert --saved-model ./saved_model \\
                            --output model.onnx

  # Then load
  net = cv2.dnn.readNetFromONNX('model.onnx')

Note: TF2 SavedModel format needs conversion to frozen graph
or ONNX for OpenCV compatibility.
"""
print(tf_info)


# =============================================================================
# 4. CAFFE FORMAT
# =============================================================================
print("\n--- 4. Caffe Format ---")

caffe_info = """
Caffe Format (.prototxt + .caffemodel):

Files:
  - .prototxt   : Network architecture (text)
  - .caffemodel : Trained weights (binary)

Loading:
  net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')

Example (Face Detection):
  proto = "deploy.prototxt"
  model = "res10_300x300_ssd_iter_140000.caffemodel"

  net = cv2.dnn.readNetFromCaffe(proto, model)

  # SSD uses different preprocessing
  blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                (104.0, 177.0, 123.0))

Note: Caffe is an older framework, but many pre-trained
models are still available in this format.
"""
print(caffe_info)


# =============================================================================
# 5. AUTO-DETECTION
# =============================================================================
print("\n--- 5. Auto-Detection ---")

auto_info = """
Auto-Detect Format:

OpenCV can auto-detect model format:
  net = cv2.dnn.readNet('model_file', 'config_file')

Examples:
  # Darknet
  net = cv2.dnn.readNet('yolo.weights', 'yolo.cfg')

  # ONNX
  net = cv2.dnn.readNet('model.onnx')

  # TensorFlow
  net = cv2.dnn.readNet('model.pb')

  # Caffe
  net = cv2.dnn.readNet('model.caffemodel', 'deploy.prototxt')
"""
print(auto_info)


# =============================================================================
# 6. BACKEND AND TARGET
# =============================================================================
print("\n--- 6. Backend and Target ---")

backend_info = """
DNN Backends and Targets:

Backends (computation engine):
  DNN_BACKEND_OPENCV      - Default, CPU
  DNN_BACKEND_CUDA        - NVIDIA GPU
  DNN_BACKEND_INFERENCE_ENGINE - Intel OpenVINO

Targets (hardware):
  DNN_TARGET_CPU          - CPU
  DNN_TARGET_CUDA         - NVIDIA GPU
  DNN_TARGET_CUDA_FP16    - NVIDIA GPU (half precision)
  DNN_TARGET_OPENCL       - OpenCL GPU
  DNN_TARGET_OPENCL_FP16  - OpenCL GPU (half precision)

Usage:
  net = cv2.dnn.readNetFromDarknet(cfg, weights)

  # CPU (default, works everywhere)
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

  # NVIDIA GPU (requires OpenCV built with CUDA)
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
"""
print(backend_info)


# =============================================================================
# 7. PREPROCESSING COMPARISON
# =============================================================================
print("\n--- 7. Preprocessing by Model ---")

preprocess_info = """
Preprocessing Parameters by Model:

YOLO (Darknet):
  blob = cv2.dnn.blobFromImage(
      img,
      scalefactor=1/255.0,
      size=(416, 416),      # or (608, 608) for better accuracy
      mean=(0, 0, 0),
      swapRB=True,
      crop=False
  )

SSD (TensorFlow/Caffe):
  blob = cv2.dnn.blobFromImage(
      img,
      scalefactor=1.0,
      size=(300, 300),
      mean=(104.0, 177.0, 123.0),
      swapRB=False,
      crop=False
  )

Classification (ImageNet models):
  blob = cv2.dnn.blobFromImage(
      img,
      scalefactor=1/255.0,  # or 0.007843 for MobileNet
      size=(224, 224),
      mean=(123.68, 116.779, 103.939),  # ImageNet mean
      swapRB=True,
      crop=False
  )
"""
print(preprocess_info)


# =============================================================================
# 8. CONVERSION WORKFLOW
# =============================================================================
print("\n--- 8. Model Conversion Workflow ---")

conversion_info = """
Recommended Conversion Workflow:

PyTorch -> ONNX -> OpenCV:
  1. Export PyTorch model to ONNX
  2. Load with cv2.dnn.readNetFromONNX()

TensorFlow -> ONNX -> OpenCV:
  1. Convert with tf2onnx
  2. Load with cv2.dnn.readNetFromONNX()

Keras -> TensorFlow -> ONNX -> OpenCV:
  1. Save Keras model as SavedModel
  2. Convert to ONNX
  3. Load with cv2.dnn.readNetFromONNX()

YOLO (Ultralytics) -> ONNX -> OpenCV:
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')
  model.export(format='onnx')

  net = cv2.dnn.readNetFromONNX('yolov8n.onnx')

Tools:
  - torch.onnx.export() - PyTorch to ONNX
  - tf2onnx - TensorFlow to ONNX
  - onnxruntime - Verify ONNX models
  - Netron - Visualize model architecture
"""
print(conversion_info)


# =============================================================================
# DEMO
# =============================================================================
def show_demo():
    """Show model loading comparison."""

    print("\n" + "=" * 60)
    print("Model Format Comparison Demo")
    print("=" * 60)

    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

    # Check which models are available
    models_status = {
        "Darknet (YOLO)": os.path.exists(os.path.join(MODEL_DIR, "yolov3-tiny.weights")),
        "ONNX": os.path.exists(os.path.join(MODEL_DIR, "yolov8n.onnx")),
        "TensorFlow": os.path.exists(os.path.join(MODEL_DIR, "model.pb")),
        "Caffe": os.path.exists(os.path.join(MODEL_DIR, "face_res10_300x300.caffemodel")),
    }

    print("\nModel Availability:")
    for fmt, available in models_status.items():
        status = "Available" if available else "Not downloaded"
        print(f"  {fmt}: {status}")

    print("\nTo use different formats:")
    print("  - Darknet: Run 01_dnn_basics.py or 02_dnn_video.py first")
    print("  - ONNX: Export from PyTorch/TensorFlow")
    print("  - See examples above for loading each format")


if __name__ == "__main__":
    show_demo()
