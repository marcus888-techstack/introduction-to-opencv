---
layout: default
title: "08: Deep Learning"
parent: Teaching Materials
nav_order: 8
permalink: /teaching-materials/08-dnn
---

# DNN From Model To Magic

Guide to loading models, blob preparation, and running inference with OpenCV DNN.

[Download PDF]({{ site.baseurl }}/teaching_materials/08-deep-learning.pdf){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 }

---

## Topics Covered

- **Model Loading** - readNet, readNetFromONNX, various formats
- **Blob Preparation** - blobFromImage parameters
- **Forward Pass** - Running inference
- **Output Processing** - Parsing network outputs
- **Model Formats** - ONNX, TensorFlow, Caffe, Darknet

---

## Tutorial Files

| File | Description |
|------|-------------|
| `01_dnn_basics.py` | Loading models, blob preparation, inference |
| `02_dnn_video.py` | Real-time video inference |
| `03_dnn_formats.py` | ONNX, TensorFlow, Caffe model formats |
