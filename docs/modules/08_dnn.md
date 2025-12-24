---
layout: default
title: "08: Deep Learning"
parent: Modules
nav_order: 8
---

# Module 8: Deep Learning (DNN)

Using deep neural networks for inference in OpenCV.

## Topics Covered

- DNN module overview
- Model loading (TensorFlow, Caffe, ONNX, Darknet)
- Blob preparation
- Inference pipeline
- Classification and detection

---

## Algorithm Explanations

### 1. DNN Module Overview

**What it does**: Runs pre-trained neural networks for inference (not training).

**DNN Inference Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                     OpenCV DNN Inference                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐ │
│   │ Load      │    │ Create    │    │ Run       │    │ Post-    │ │
│   │ Model     │───▶│ Blob      │───▶│ Inference │───▶│ Process  │ │
│   │           │    │           │    │           │    │          │ │
│   └───────────┘    └───────────┘    └───────────┘    └──────────┘ │
│        │                │                │                │        │
│        ▼                ▼                ▼                ▼        │
│   .weights/.pb     blobFromImage    net.forward()    Parse        │
│   .cfg/.onnx       (normalize,      (GPU/CPU)        outputs      │
│                    resize)                                         │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │          OpenCV handles framework differences               │  │
│   │   TensorFlow ←→ Caffe ←→ ONNX ←→ Darknet ←→ PyTorch        │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Supported Frameworks**:
| Framework | Model File | Config File |
|-----------|------------|-------------|
| TensorFlow | `.pb` | `.pbtxt` (optional) |
| Caffe | `.caffemodel` | `.prototxt` |
| Darknet/YOLO | `.weights` | `.cfg` |
| ONNX | `.onnx` | - |
| PyTorch | via ONNX export | - |

**Backends**:
| Backend | Target | Description |
|---------|--------|-------------|
| `DNN_BACKEND_OPENCV` | CPU | Default, pure OpenCV |
| `DNN_BACKEND_CUDA` | GPU | NVIDIA GPU acceleration |
| `DNN_BACKEND_INFERENCE_ENGINE` | CPU/GPU | Intel OpenVINO |

---

### 2. Blob Format

**What it does**: Converts image to neural network input format.

**Blob Transformation Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                  blobFromImage() Transformation                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Image (HWC)                    Output Blob (NCHW)          │
│   OpenCV format                        Neural network format       │
│                                                                     │
│   ┌───────────────┐                    ┌─────────────────────┐     │
│   │ ┌───────────┐ │                    │ Batch 0             │     │
│   │ │   Blue    │ │                    │ ┌───┬───┬───┐       │     │
│   │ │  Channel  │ │                    │ │ R │ G │ B │       │     │
│   │ ├───────────┤ │   blobFromImage()  │ │   │   │   │       │     │
│   │ │   Green   │ │   ───────────────▶ │ │ C │ C │ C │       │     │
│   │ │  Channel  │ │   • resize         │ │ h │ h │ h │       │     │
│   │ ├───────────┤ │   • scale          │ │ a │ a │ a │       │     │
│   │ │   Red     │ │   • mean subtract  │ │ n │ n │ n │       │     │
│   │ │  Channel  │ │   • swap R↔B       │ │   │   │   │       │     │
│   │ └───────────┘ │                    │ └───┴───┴───┘       │     │
│   │    H × W × 3  │                    │   1 × 3 × H × W     │     │
│   └───────────────┘                    └─────────────────────┘     │
│                                                                     │
│   Shape: (480, 640, 3)       →         Shape: (1, 3, 224, 224)     │
│   Range: [0, 255]            →         Range: [0.0, 1.0] or norm   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**NCHW Format**:
```
N = Batch size
C = Channels (3 for RGB)
H = Height
W = Width

Shape: (1, 3, 224, 224) for typical ImageNet input
```

**blobFromImage Parameters**:
```python
blob = cv2.dnn.blobFromImage(
    image,          # Input image (BGR)
    scalefactor,    # Pixel value scaling (e.g., 1/255)
    size,           # Output dimensions (width, height)
    mean,           # Mean subtraction values (B, G, R)
    swapRB,         # Swap R and B channels (BGR→RGB)
    crop            # Center crop to size
)
```

**Common Preprocessing**:
| Model | scalefactor | size | mean | swapRB |
|-------|-------------|------|------|--------|
| ImageNet | 1/255 | (224, 224) | (0, 0, 0) | True |
| VGG | 1.0 | (224, 224) | (103.939, 116.779, 123.68) | False |
| SSD | 1.0 | (300, 300) | (104, 177, 123) | False |
| YOLO | 1/255 | (416, 416) | (0, 0, 0) | True |

---

### 3. Inference Pipeline

**Step-by-Step**:

```python
# 1. Load model
net = cv2.dnn.readNet('model.weights', 'model.cfg')

# 2. Set backend/target (optional)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 3. Prepare input
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True)

# 4. Set input
net.setInput(blob)

# 5. Forward pass
output = net.forward()  # Single output
# or
outputs = net.forward(output_layer_names)  # Multiple outputs

# 6. Post-process results
```

**Getting Output Layer Names**:
```python
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```

---

### 4. Classification

**What it does**: Assigns image to one of N categories.

**Classification Pipeline**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Image Classification                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Image         Neural Network           Output Vector       │
│                                                                     │
│   ┌───────────┐      ┌─────────────┐      ┌───────────────────┐   │
│   │   🐱     │      │ ┌─────────┐ │      │  cat:     0.92   │   │
│   │   Cat     │  ──▶ │ │ Conv   │ │  ──▶ │  dog:     0.05   │   │
│   │  Image    │      │ ├─────────┤ │      │  bird:    0.02   │   │
│   │           │      │ │ Conv   │ │      │  car:     0.01   │   │
│   └───────────┘      │ ├─────────┤ │      │  ...             │   │
│                       │ │  FC    │ │      │                   │   │
│   224×224×3          │ ├─────────┤ │      │  N classes        │   │
│                       │ │Softmax │ │      │  (probabilities)  │   │
│                       │ └─────────┘ │      └───────────────────┘   │
│                       └─────────────┘                              │
│                                                                     │
│   argmax() → class_id = 0 (cat)                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Output**: Probability vector of shape `(1, N)`

**Processing**:
```python
blob = cv2.dnn.blobFromImage(image, 1/255.0, (224, 224), swapRB=True)
net.setInput(blob)
predictions = net.forward()

# Get top prediction
class_id = np.argmax(predictions[0])
confidence = predictions[0][class_id]
```

**Softmax** (if not applied in model):
```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

---

### 5. Object Detection (YOLO)

**YOLO Detection Concept**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOLO: You Only Look Once                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input Image               Grid Division        Per-Cell Output   │
│                                                                     │
│   ┌─────────────┐          ┌───┬───┬───┐       Each cell predicts: │
│   │   🚗       │          │   │ 🚗│   │       • B bounding boxes  │
│   │  ┌───┐     │   ───▶   ├───┼───┼───┤       • Confidence scores │
│   │  │car│     │   S×S    │   │   │   │       • C class probs     │
│   │  └───┘     │   grid   ├───┼───┼───┤                            │
│   │     🐕    │          │   │   │ 🐕│                            │
│   └─────────────┘          └───┴───┴───┘                            │
│                                                                     │
│   Single forward pass → detect all objects at once (fast!)         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**YOLO Output Vector**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Detection Output Format                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Each detection = [cx, cy, w, h, obj, c1, c2, c3, ..., cN]        │
│                                                                     │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐          │
│   │ cx  │ cy  │  w  │  h  │ obj │ c1  │ c2  │ c3  │ ... │          │
│   └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴─────┴─────┴─────┘          │
│      │     │     │     │     │     │                                │
│      │     │     │     │     │     └── Class probabilities          │
│      │     │     │     │     │         (person, car, dog, ...)      │
│      │     │     │     │     │                                      │
│      │     │     │     │     └── Objectness (P(object))             │
│      │     │     │     │                                            │
│      │     │     └─────┴── Box size (normalized 0-1)                │
│      │     │                                                        │
│      └─────┴── Box center (normalized 0-1)                          │
│                                                                     │
│   Final confidence = objectness × class_probability                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Output Structure** (per detection):
```
[center_x, center_y, width, height, objectness, class_1_prob, class_2_prob, ...]
```

**Processing**:
```python
for detection in output:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id] * detection[4]  # objectness × class_prob

    if confidence > threshold:
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)

        x = center_x - w // 2
        y = center_y - h // 2
```

**Non-Maximum Suppression**:
```python
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
```

---

### 6. SSD Detection Output

**SSD vs YOLO Output Comparison**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Detection Output Formats                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   YOLO Output:                                                      │
│   ┌────────────────────────────────────────────────┐               │
│   │ [cx, cy, w, h, obj, class_probs...]            │  Relative     │
│   │  └──normalized 0-1──┘                          │  coords       │
│   └────────────────────────────────────────────────┘               │
│                                                                     │
│   SSD Output:                                                       │
│   ┌────────────────────────────────────────────────┐               │
│   │ [batch, class, conf, x1, y1, x2, y2]           │  Corner       │
│   │                       └──normalized 0-1──┘     │  coords       │
│   └────────────────────────────────────────────────┘               │
│                                                                     │
│   Key Differences:                                                  │
│   • YOLO: center + width/height                                    │
│   • SSD: top-left + bottom-right corners                           │
│   • Both normalized to [0, 1]                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Output Format**: `(1, 1, N, 7)` where each detection is:
```
[batch_id, class_id, confidence, x1, y1, x2, y2]
```
Coordinates are normalized [0, 1].

**Processing**:
```python
for detection in output[0, 0]:
    confidence = detection[2]
    if confidence > threshold:
        class_id = int(detection[1])
        x1 = int(detection[3] * width)
        y1 = int(detection[4] * height)
        x2 = int(detection[5] * width)
        y2 = int(detection[6] * height)
```

---

### 7. Performance Optimization

**Profiling**:
```python
t, _ = net.getPerfProfile()
time_ms = t * 1000 / cv2.getTickFrequency()
```

**Optimization Strategies**:

1. **Use GPU**:
   ```python
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
   ```

2. **Reduce Input Size**:
   - Smaller blobs = faster inference
   - Trade-off with accuracy

3. **Batch Processing**:
   ```python
   blob = cv2.dnn.blobFromImages(images, ...)  # Multiple images
   ```

4. **Use FP16** (if supported):
   ```python
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
   ```

5. **Model Optimization**:
   - Quantization (INT8)
   - Pruning
   - Knowledge distillation

---

### 8. Common Architectures

**Classification**:
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| MobileNet | Small | Fast | Good | Mobile/embedded |
| ResNet | Large | Medium | Excellent | High accuracy |
| EfficientNet | Medium | Medium | Best | Balanced |

**Detection**:
| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLO v3-v8 | Fast | Good | Real-time |
| SSD | Fast | Good | Real-time |
| Faster R-CNN | Slow | Excellent | High accuracy |

**Segmentation**:
| Model | Type | Use Case |
|-------|------|----------|
| FCN | Semantic | General |
| U-Net | Instance | Medical |
| DeepLab | Semantic | High quality |

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_dnn_basics.py` | Loading models, blob preparation, inference |

---

## Key Functions Reference

| Function | Description |
|----------|-------------|
| `cv2.dnn.readNet()` | Auto-detect and load model |
| `cv2.dnn.readNetFromDarknet()` | Load Darknet/YOLO |
| `cv2.dnn.readNetFromTensorflow()` | Load TensorFlow |
| `cv2.dnn.readNetFromCaffe()` | Load Caffe |
| `cv2.dnn.readNetFromONNX()` | Load ONNX |
| `cv2.dnn.blobFromImage()` | Create input blob |
| `net.setInput()` | Set network input |
| `net.forward()` | Run inference |
| `net.setPreferableBackend()` | Set computation backend |
| `net.setPreferableTarget()` | Set target device |
| `cv2.dnn.NMSBoxes()` | Non-max suppression |

---

## Further Reading

- [DNN Tutorial](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [Model Zoo](https://github.com/opencv/opencv/tree/master/samples/dnn)
