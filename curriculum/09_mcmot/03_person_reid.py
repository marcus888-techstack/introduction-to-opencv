"""
Module 10: MCMOT - Person Re-Identification (Re-ID)
====================================================
CORE MODULE: Feature extraction and appearance matching for person tracking.

Official Docs: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
Model Source: https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtu

Topics Covered:
1. Re-ID Concepts
2. Feature Extraction with DNN
3. Cosine Similarity/Distance
4. Gallery-Query Matching
5. Re-ID Pipeline
6. Practical Applications
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_sample_path

print("=" * 60)
print("Module 10: Person Re-Identification (Re-ID)")
print("=" * 60)


# =============================================================================
# 1. RE-ID CONCEPTS
# =============================================================================
print("\n--- 1. Re-ID Concepts ---")

reid_concepts = """
Person Re-Identification (Re-ID):

Goal: Match the SAME person across different images, views, or cameras.

Why It's Needed:
  - Track through occlusion (person hidden, then reappears)
  - Prevent ID switches when paths cross
  - Match across multiple cameras
  - Re-identify after detection gaps

How It Works:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   1. FEATURE EXTRACTION                                                 │
│   ┌─────────┐        ┌─────────────┐        ┌────────────────┐         │
│   │ Person  │  ────> │  Re-ID CNN  │  ────> │ Feature Vector │         │
│   │  Crop   │        │  (128x256)  │        │   (512-dim)    │         │
│   └─────────┘        └─────────────┘        └────────────────┘         │
│                                                                          │
│   2. SIMILARITY COMPARISON                                              │
│   ┌────────────────┐                                                    │
│   │ feat_A ──────────┐                                                  │
│   │                  │── cosine_distance ──> 0.15 (SAME person)        │
│   │ feat_B ──────────┘                                                  │
│   └────────────────┘                                                    │
│                                                                          │
│   3. MATCHING DECISION                                                  │
│      distance < threshold (0.5) → Same person                           │
│      distance > threshold       → Different persons                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Feature Space Properties:
  - Same person → Features cluster together (low distance)
  - Different persons → Features far apart (high distance)
  - Robust to: pose changes, lighting, partial occlusion
"""
print(reid_concepts)


# =============================================================================
# 2. LOADING RE-ID MODEL
# =============================================================================
print("\n--- 2. Loading Re-ID Model ---")

model_info = """
Person Re-ID Model (OpenCV Zoo):

Model: person_reid_youtu_2021nov.onnx
  - Input: 128 x 256 RGB image (person crop)
  - Output: 512-dimensional feature vector
  - Based on: OSNet architecture
  - Trained on: Large-scale person re-identification datasets

Usage with OpenCV DNN:
  reid_net = cv2.dnn.readNetFromONNX("person_reid_youtu_2021nov.onnx")
  blob = cv2.dnn.blobFromImage(crop, 1/255.0, (128, 256), ...)
  reid_net.setInput(blob)
  features = reid_net.forward()
"""
print(model_info)


def load_reid_model():
    """
    Load Person Re-ID model.

    Returns:
        reid_net: OpenCV DNN network for Re-ID
    """
    model_path = get_sample_path("person_reid_youtu_2021nov.onnx")

    if not os.path.exists(model_path):
        print(f"  Re-ID model not found: {model_path}")
        print("  Run: python download_samples.py")
        return None

    print(f"  Loading Re-ID model: {model_path}")
    reid_net = cv2.dnn.readNetFromONNX(model_path)

    # Set backend
    reid_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    reid_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("  Re-ID model loaded successfully")
    return reid_net


# Load model
reid_net = load_reid_model()


# =============================================================================
# 3. FEATURE EXTRACTION
# =============================================================================
print("\n--- 3. Feature Extraction ---")

extraction_info = """
Feature Extraction Pipeline:

1. Preprocess person crop:
   - Resize to 128 x 256 (width x height)
   - Normalize to [0, 1] range
   - Apply ImageNet mean subtraction (optional)
   - Swap BGR to RGB

2. Create blob:
   blob = cv2.dnn.blobFromImage(
       crop,
       scalefactor=1/255.0,       # Normalize to [0, 1]
       size=(128, 256),           # Model input size
       mean=(0.485, 0.456, 0.406),# ImageNet mean (RGB)
       swapRB=True                # BGR to RGB
   )

3. Forward pass:
   reid_net.setInput(blob)
   features = reid_net.forward()  # Shape: (1, 512)

4. L2 Normalize:
   features = features / np.linalg.norm(features)
"""
print(extraction_info)


def extract_features(person_crop, reid_net):
    """
    Extract Re-ID features from a person crop.

    Args:
        person_crop: BGR image of person (any size, will be resized)
        reid_net: OpenCV DNN Re-ID network

    Returns:
        features: 512-dimensional normalized feature vector
    """
    if reid_net is None:
        # Return random features for demo without model
        features = np.random.randn(512).astype(np.float32)
        return features / np.linalg.norm(features)

    # Target size for Re-ID model
    target_size = (128, 256)  # width x height

    # Resize if needed
    if person_crop.shape[:2] != (256, 128):
        crop_resized = cv2.resize(person_crop, target_size)
    else:
        crop_resized = person_crop

    # Create blob with preprocessing
    # Note: The model expects RGB with specific normalization
    blob = cv2.dnn.blobFromImage(
        crop_resized,
        scalefactor=1.0 / 255.0,
        size=target_size,
        mean=(0.0, 0.0, 0.0),  # No mean subtraction for this model
        swapRB=True,
        crop=False
    )

    # Forward pass
    reid_net.setInput(blob)
    features = reid_net.forward()

    # L2 normalize the features
    features = features.flatten()
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm

    return features


def extract_features_batch(person_crops, reid_net):
    """
    Extract Re-ID features from multiple person crops (batch processing).

    Args:
        person_crops: List of BGR images
        reid_net: OpenCV DNN Re-ID network

    Returns:
        List of feature vectors
    """
    if not person_crops:
        return []

    features_list = []
    for crop in person_crops:
        features = extract_features(crop, reid_net)
        features_list.append(features)

    return features_list


# =============================================================================
# 4. COSINE SIMILARITY & DISTANCE
# =============================================================================
print("\n--- 4. Cosine Similarity & Distance ---")

similarity_info = """
Cosine Similarity - Measuring Feature Similarity:

Formula:
                        A · B           Σ(A[i] × B[i])
    similarity(A, B) = ─────── = ─────────────────────────
                       |A||B|    √(Σ A[i]²) × √(Σ B[i]²)

    Range: [-1, 1]
    - 1.0: Identical direction (most similar)
    - 0.0: Orthogonal (no similarity)
    - -1.0: Opposite direction

Cosine Distance:
    distance(A, B) = 1 - similarity(A, B)

    Range: [0, 2]
    - 0.0: Identical features
    - 1.0: Orthogonal
    - 2.0: Opposite

For L2-normalized features (unit vectors):
    distance = 1 - dot(A, B)

Typical Thresholds:
    distance < 0.3  →  Very confident same person
    distance < 0.5  →  Likely same person
    distance > 0.7  →  Likely different persons
"""
print(similarity_info)


def cosine_similarity(feat1, feat2):
    """
    Compute cosine similarity between two feature vectors.

    Args:
        feat1: First feature vector (assumed L2-normalized)
        feat2: Second feature vector (assumed L2-normalized)

    Returns:
        Similarity value in range [-1, 1]
    """
    return np.dot(feat1, feat2)


def cosine_distance(feat1, feat2):
    """
    Compute cosine distance between two feature vectors.

    Args:
        feat1: First feature vector (assumed L2-normalized)
        feat2: Second feature vector (assumed L2-normalized)

    Returns:
        Distance value in range [0, 2]
    """
    return 1.0 - cosine_similarity(feat1, feat2)


def euclidean_distance(feat1, feat2):
    """
    Compute Euclidean distance between two feature vectors.

    Args:
        feat1: First feature vector
        feat2: Second feature vector

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(feat1 - feat2)


# Demo similarity calculation
print("\nSimilarity Demo:")
feat_a = np.array([0.8, 0.2, 0.1, 0.4, 0.3])
feat_a = feat_a / np.linalg.norm(feat_a)

feat_b = np.array([0.75, 0.25, 0.15, 0.38, 0.32])  # Similar to A
feat_b = feat_b / np.linalg.norm(feat_b)

feat_c = np.array([-0.3, 0.7, -0.5, 0.2, 0.8])  # Different from A
feat_c = feat_c / np.linalg.norm(feat_c)

print(f"  Distance(A, B): {cosine_distance(feat_a, feat_b):.4f} (Similar)")
print(f"  Distance(A, C): {cosine_distance(feat_a, feat_c):.4f} (Different)")


# =============================================================================
# 5. GALLERY-QUERY MATCHING
# =============================================================================
print("\n--- 5. Gallery-Query Matching ---")

gallery_info = """
Gallery-Query Re-ID Matching:

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  GALLERY (Known Persons)              QUERY (New Detection)             │
│  ┌──────────────────────┐             ┌──────────────────────┐         │
│  │ ID=1: features_1     │             │ query_features       │         │
│  │ ID=2: features_2     │             │                      │         │
│  │ ID=3: features_3     │   Match?    │ Who is this?         │         │
│  │ ...                  │ <────────── │                      │         │
│  └──────────────────────┘             └──────────────────────┘         │
│                                                                          │
│  Matching Algorithm:                                                    │
│  1. Compute distance to each gallery entry                              │
│  2. Find minimum distance                                               │
│  3. If min_distance < threshold → Match to that ID                      │
│  4. If min_distance >= threshold → Create new ID                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
"""
print(gallery_info)


class PersonGallery:
    """
    Gallery of known persons for Re-ID matching.
    """

    def __init__(self, distance_threshold=0.5, feature_dim=512):
        """
        Initialize gallery.

        Args:
            distance_threshold: Max distance to consider a match
            feature_dim: Dimension of feature vectors
        """
        self.distance_threshold = distance_threshold
        self.feature_dim = feature_dim
        self.gallery = {}  # {person_id: features}
        self.next_id = 1

    def add_person(self, features, person_id=None):
        """
        Add a new person to the gallery.

        Args:
            features: Feature vector for the person
            person_id: Optional specific ID (auto-generates if None)

        Returns:
            Assigned person ID
        """
        if person_id is None:
            person_id = self.next_id
            self.next_id += 1

        self.gallery[person_id] = features.copy()
        return person_id

    def update_person(self, person_id, new_features, alpha=0.1):
        """
        Update person features with exponential moving average.

        Args:
            person_id: ID of person to update
            new_features: New feature vector
            alpha: Weight for new features (0 = keep old, 1 = replace)
        """
        if person_id in self.gallery:
            old_features = self.gallery[person_id]
            updated = (1 - alpha) * old_features + alpha * new_features
            # Re-normalize
            self.gallery[person_id] = updated / np.linalg.norm(updated)

    def find_match(self, query_features):
        """
        Find matching person in gallery.

        Args:
            query_features: Feature vector of query person

        Returns:
            (matched_id, distance) or (None, None) if no match
        """
        if not self.gallery:
            return None, None

        best_id = None
        best_distance = float('inf')

        for person_id, gallery_features in self.gallery.items():
            dist = cosine_distance(query_features, gallery_features)
            if dist < best_distance:
                best_distance = dist
                best_id = person_id

        if best_distance < self.distance_threshold:
            return best_id, best_distance
        else:
            return None, best_distance

    def match_or_create(self, query_features):
        """
        Match to existing person or create new entry.

        Args:
            query_features: Feature vector of query person

        Returns:
            (person_id, is_new, distance)
        """
        matched_id, distance = self.find_match(query_features)

        if matched_id is not None:
            # Update existing entry
            self.update_person(matched_id, query_features)
            return matched_id, False, distance
        else:
            # Create new entry
            new_id = self.add_person(query_features)
            # For new entries, distance is 0 (perfect match to self)
            return new_id, True, 0.0

    def get_all_ids(self):
        """Get list of all person IDs in gallery."""
        return list(self.gallery.keys())

    def size(self):
        """Get number of persons in gallery."""
        return len(self.gallery)


# Demo gallery matching
print("\nGallery Demo:")
gallery = PersonGallery(distance_threshold=0.5)

# Add some persons
feat1 = np.random.randn(512).astype(np.float32)
feat1 /= np.linalg.norm(feat1)
id1 = gallery.add_person(feat1)
print(f"  Added person ID={id1}")

feat2 = np.random.randn(512).astype(np.float32)
feat2 /= np.linalg.norm(feat2)
id2 = gallery.add_person(feat2)
print(f"  Added person ID={id2}")

# Query with similar features to person 1
query_similar = feat1 + 0.1 * np.random.randn(512).astype(np.float32)
query_similar /= np.linalg.norm(query_similar)

matched_id, is_new, dist = gallery.match_or_create(query_similar)
print(f"  Query (similar to ID=1): matched={matched_id}, new={is_new}, dist={dist:.4f}")

# Query with different features
query_different = np.random.randn(512).astype(np.float32)
query_different /= np.linalg.norm(query_different)

matched_id, is_new, dist = gallery.match_or_create(query_different)
print(f"  Query (different): matched={matched_id}, new={is_new}, dist={dist:.4f}")


# =============================================================================
# 6. COMPLETE RE-ID PIPELINE
# =============================================================================
print("\n--- 6. Complete Re-ID Pipeline ---")


class PersonReID:
    """
    Complete Person Re-ID system.
    """

    def __init__(self, reid_net, distance_threshold=0.5):
        """
        Initialize Re-ID system.

        Args:
            reid_net: OpenCV DNN Re-ID network
            distance_threshold: Matching threshold
        """
        self.reid_net = reid_net
        self.gallery = PersonGallery(distance_threshold=distance_threshold)

    def extract_features(self, person_crop):
        """Extract features from person crop."""
        return extract_features(person_crop, self.reid_net)

    def identify(self, person_crop):
        """
        Identify a person from their image crop.

        Args:
            person_crop: BGR image of person

        Returns:
            (person_id, is_new, confidence)
        """
        features = self.extract_features(person_crop)
        person_id, is_new, distance = self.gallery.match_or_create(features)

        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0, 1 - distance)

        return person_id, is_new, confidence

    def identify_batch(self, person_crops):
        """
        Identify multiple persons.

        Args:
            person_crops: List of BGR images

        Returns:
            List of (person_id, is_new, confidence) tuples
        """
        results = []
        for crop in person_crops:
            result = self.identify(crop)
            results.append(result)
        return results

    def get_gallery_size(self):
        """Get number of known persons."""
        return self.gallery.size()


# =============================================================================
# 7. VISUALIZATION & REAL DATA LOADING
# =============================================================================
print("\n--- 7. Visualization ---")


def load_yolo_detector():
    """Load YOLO detector for person detection."""
    weights_path = get_sample_path("yolov4-tiny.weights")
    cfg_path = get_sample_path("yolov4-tiny.cfg")

    if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
        print("  YOLO model not found. Run: python download_samples.py")
        return None, None

    net = cv2.dnn.readNet(weights_path, cfg_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers


def detect_persons_in_frame(frame, net, output_layers, conf_threshold=0.5):
    """Detect persons in a frame using YOLO."""
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only keep person class (class_id == 0)
            if class_id == 0 and confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append(boxes[i])

    return result


def crop_person(frame, box, target_size=(128, 256)):
    """Crop and resize person from frame."""
    x, y, w, h = box
    h_img, w_img = frame.shape[:2]

    # Clamp to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    # Resize maintaining aspect ratio with padding
    crop_h, crop_w = crop.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    resized = cv2.resize(crop, (new_w, new_h))

    # Create padded output
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result


def extract_person_crops_from_video(video_path, num_crops=10, skip_frames=10):
    """
    Extract person crops from video using YOLO detection.

    Args:
        video_path: Path to video file
        num_crops: Maximum number of crops to extract
        skip_frames: Frames to skip between extractions

    Returns:
        List of (crop, frame_idx, box) tuples
    """
    print(f"  Loading video: {video_path}")

    # Load YOLO
    net, output_layers = load_yolo_detector()
    if net is None:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return []

    crops = []
    frame_idx = 0

    while len(crops) < num_crops:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % skip_frames != 0:
            continue

        # Detect persons
        boxes = detect_persons_in_frame(frame, net, output_layers)

        for box in boxes:
            crop = crop_person(frame, box)
            if crop is not None:
                crops.append((crop, frame_idx, box, frame.copy()))
                if len(crops) >= num_crops:
                    break

    cap.release()
    print(f"  Extracted {len(crops)} person crops from video")

    return crops


def create_test_person_crops():
    """
    Get real person crops from sample video.
    Falls back to synthetic if video unavailable.
    """
    # Try to use real video
    video_path = get_sample_path("vtest.avi")

    if os.path.exists(video_path):
        crop_data = extract_person_crops_from_video(video_path, num_crops=8, skip_frames=15)
        if crop_data:
            return [c[0] for c in crop_data]  # Return just the crops

    # Fallback: synthetic crops
    print("  Using synthetic test crops (video not available)")
    crops = []
    colors = [
        (50, 100, 150), (80, 60, 120), (100, 150, 50),
        (150, 80, 80), (60, 130, 130)
    ]

    for i, color in enumerate(colors):
        crop = np.ones((256, 128, 3), dtype=np.uint8) * 200
        cv2.rectangle(crop, (20, 60), (108, 240), color, -1)
        cv2.circle(crop, (64, 40), 30, (200, 180, 160), -1)
        crops.append(crop)

    return crops


def visualize_matching(crops, reid_system):
    """
    Visualize Re-ID matching results.

    Args:
        crops: List of person crops
        reid_system: PersonReID instance

    Returns:
        Visualization image
    """
    # Process each crop
    results = []
    for crop in crops:
        person_id, is_new, confidence = reid_system.identify(crop)
        results.append((crop, person_id, is_new, confidence))

    # Create visualization
    crop_h, crop_w = 256, 128
    padding = 10
    n_crops = len(crops)

    vis_w = n_crops * (crop_w + padding) + padding
    vis_h = crop_h + 80

    vis = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255

    # Color palette for IDs
    id_colors = {}

    for i, (crop, person_id, is_new, confidence) in enumerate(results):
        x = padding + i * (crop_w + padding)
        y = 40

        # Assign color to ID
        if person_id not in id_colors:
            np.random.seed(person_id * 123)
            id_colors[person_id] = tuple(np.random.randint(50, 200, 3).tolist())

        color = id_colors[person_id]

        # Draw crop
        vis[y:y+crop_h, x:x+crop_w] = crop

        # Draw border with ID color
        cv2.rectangle(vis, (x-2, y-2), (x+crop_w+2, y+crop_h+2), color, 3)

        # Draw ID and confidence
        label = f"ID={person_id}"
        if is_new:
            label += " (NEW)"

        cv2.putText(vis, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(vis, f"Conf: {confidence:.2f}", (x, y + crop_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Title
    cv2.putText(vis, "Person Re-ID Matching", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return vis


def visualize_distance_matrix(features_list, labels=None):
    """
    Visualize pairwise distances between features.

    Args:
        features_list: List of feature vectors
        labels: Optional labels for each feature

    Returns:
        Distance matrix visualization
    """
    n = len(features_list)

    if labels is None:
        labels = [f"P{i+1}" for i in range(n)]

    # Compute distance matrix
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = cosine_distance(features_list[i], features_list[j])

    # Create visualization
    cell_size = 60
    margin = 80
    vis_size = margin + n * cell_size

    vis = np.ones((vis_size, vis_size, 3), dtype=np.uint8) * 255

    # Draw cells
    for i in range(n):
        for j in range(n):
            x = margin + j * cell_size
            y = margin + i * cell_size

            # Color based on distance (green = low, red = high)
            dist = distances[i, j]
            if dist < 0.3:
                color = (100, 200, 100)  # Green
            elif dist < 0.5:
                color = (150, 200, 150)  # Light green
            elif dist < 0.7:
                color = (200, 200, 150)  # Yellow
            else:
                color = (150, 150, 200)  # Red-ish

            cv2.rectangle(vis, (x, y), (x + cell_size - 2, y + cell_size - 2),
                         color, -1)
            cv2.rectangle(vis, (x, y), (x + cell_size - 2, y + cell_size - 2),
                         (100, 100, 100), 1)

            # Draw distance value
            cv2.putText(vis, f"{dist:.2f}",
                       (x + 10, y + cell_size // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw labels
    for i, label in enumerate(labels):
        x = margin + i * cell_size + 15
        y = margin - 10
        cv2.putText(vis, label, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        x = 10
        y = margin + i * cell_size + cell_size // 2 + 5
        cv2.putText(vis, label, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Title
    cv2.putText(vis, "Distance Matrix", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return vis, distances


# =============================================================================
# DEMO
# =============================================================================
def run_video_reid_demo():
    """
    Run Re-ID demo on video - showing real-time person identification.
    """
    print("\n  Running video Re-ID demo...")

    # Load video
    video_path = get_sample_path("vtest.avi")
    if not os.path.exists(video_path):
        print(f"  Video not found: {video_path}")
        print("  Run: python download_samples.py")
        return []

    # Load YOLO detector
    yolo_net, yolo_layers = load_yolo_detector()
    if yolo_net is None:
        return []

    # Create Re-ID system
    reid_system = PersonReID(reid_net, distance_threshold=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return []

    result_frames = []
    frame_count = 0
    max_frames = 150

    # Color palette for person IDs
    def get_color(person_id):
        np.random.seed(person_id * 123 + 42)
        return tuple(np.random.randint(50, 255, 3).tolist())

    print(f"  Processing video...")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect persons
        boxes = detect_persons_in_frame(frame, yolo_net, yolo_layers, conf_threshold=0.4)

        vis_frame = frame.copy()

        # Process each detected person
        for box in boxes:
            x, y, w, h = box

            # Crop person
            crop = crop_person(frame, box)
            if crop is None:
                continue

            # Identify using Re-ID
            person_id, is_new, confidence = reid_system.identify(crop)

            # Get color for this person
            color = get_color(person_id)

            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)

            # Draw ID label
            label = f"ID:{person_id}"
            if is_new:
                label += " (NEW)"

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x, y - text_h - 10), (x + text_w + 5, y), color, -1)
            cv2.putText(vis_frame, label, (x + 2, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw confidence bar
            bar_width = int(w * confidence)
            cv2.rectangle(vis_frame, (x, y + h + 2), (x + bar_width, y + h + 8), color, -1)

        # Add info overlay
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"Persons in Gallery: {reid_system.get_gallery_size()}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"Detections: {len(boxes)}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        result_frames.append(vis_frame)

    cap.release()
    print(f"  Processed {frame_count} frames")
    print(f"  Total unique persons identified: {reid_system.get_gallery_size()}")

    return result_frames


def show_demo():
    """Display Re-ID demonstrations."""
    print("\n" + "=" * 60)
    print("Running Re-ID Demo...")
    print("=" * 60)

    # Run video-based Re-ID demo
    result_frames = run_video_reid_demo()

    if result_frames:
        print("\nShowing video Re-ID demo...")
        print("Press any key to continue, ESC to exit")

        for frame in result_frames:
            cv2.imshow("Person Re-ID on Video", frame)
            key = cv2.waitKey(30)
            if key == 27:  # ESC
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nCould not run video demo. Showing static demo instead...")

        # Fallback: static crops demo
        reid_system = PersonReID(reid_net, distance_threshold=0.5)
        crops = create_test_person_crops()

        if crops:
            print(f"  Got {len(crops)} person crops")
            vis = visualize_matching(crops, reid_system)
            cv2.imshow("Re-ID Matching", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running Person Re-ID demonstrations...")
    print("=" * 60)
    show_demo()
