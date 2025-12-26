"""
Extra Module: Face Recognition (opencv-contrib)
================================================
Face detection, recognition, and analysis using contrib modules.

Note: Requires opencv-contrib-python

Topics Covered:
1. Face Detection with DNN
2. Face Landmark Detection
3. Face Recognition (Eigenfaces, Fisherfaces, LBPH)
4. Face Alignment
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Extra Module: Face Recognition")
print("=" * 60)


# =============================================================================
# 1. CHECK OPENCV CONTRIB
# =============================================================================
print("\n--- 1. Checking OpenCV Installation ---")

try:
    face_module = cv2.face
    print("OpenCV Face module: Available")
except AttributeError:
    print("OpenCV Face module: NOT Available")
    print("Install with: pip install opencv-contrib-python")
    face_module = None


# =============================================================================
# 2. DNN FACE DETECTION
# =============================================================================
print("\n--- 2. DNN Face Detection ---")

dnn_info = """
DNN Face Detection (recommended approach):

Model: SSD with ResNet10 backbone

Files needed:
  - opencv_face_detector_uint8.pb
  - opencv_face_detector.pbtxt

Download from:
  https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

Usage:
  net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
  blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
  net.setInput(blob)
  detections = net.forward()
"""
print(dnn_info)


def load_face_image():
    """Load a real face image for demo or create synthetic fallback."""
    # Try to load real face images
    # OpenCV samples: lena.jpg (classic face), messi5.jpg (Messi face image)
    for sample in ["lena.jpg", "messi5.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return cv2.resize(img, (300, 300))

    # Fallback: Create synthetic face-like image
    print("No sample face image found. Using synthetic face.")
    print("Run: python curriculum/sample_data/download_samples.py")
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (200, 180, 160)

    # Face
    cv2.ellipse(img, (150, 150), (80, 100), 0, 0, 360, (180, 160, 140), -1)

    # Eyes
    cv2.ellipse(img, (110, 120), (20, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (190, 120), (20, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (110, 120), 8, (50, 50, 50), -1)
    cv2.circle(img, (190, 120), 8, (50, 50, 50), -1)

    # Eyebrows
    cv2.line(img, (80, 95), (140, 100), (80, 60, 40), 4)
    cv2.line(img, (160, 100), (220, 95), (80, 60, 40), 4)

    # Nose
    pts = np.array([[150, 130], [135, 175], [165, 175]], np.int32)
    cv2.polylines(img, [pts], True, (150, 130, 110), 2)

    # Mouth
    cv2.ellipse(img, (150, 200), (30, 15), 0, 0, 180, (120, 80, 80), -1)

    return img


face_img = load_face_image()


# =============================================================================
# 3. FACE RECOGNIZER
# =============================================================================
print("\n--- 3. Face Recognition Algorithms ---")

if face_module:
    recognizer_info = """
Face Recognition Algorithms:

1. Eigenfaces (PCA):
   recognizer = cv2.face.EigenFaceRecognizer_create(num_components, threshold)
   - Uses PCA for dimensionality reduction
   - Fast but sensitive to lighting

2. Fisherfaces (LDA):
   recognizer = cv2.face.FisherFaceRecognizer_create(num_components, threshold)
   - Uses Linear Discriminant Analysis
   - Better than Eigenfaces for varying lighting

3. LBPH (Local Binary Patterns Histograms):
   recognizer = cv2.face.LBPHFaceRecognizer_create(radius, neighbors, grid_x, grid_y, threshold)
   - Texture-based approach
   - Robust to lighting changes
   - Can be updated with new faces

Common Methods:
   recognizer.train(images, labels)   # Train with labeled faces
   label, confidence = recognizer.predict(face)  # Predict identity
   recognizer.update(images, labels)  # LBPH only: add new faces
   recognizer.save('model.yml')       # Save model
   recognizer.read('model.yml')       # Load model
"""
    print(recognizer_info)

    # Demo: Create and train LBPH recognizer
    try:
        lbph = cv2.face.LBPHFaceRecognizer_create()
        print("LBPH Recognizer created successfully")

        # Create dummy training data
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        faces = [gray, cv2.flip(gray, 1)]  # Original and flipped
        labels = np.array([0, 0])  # Same person

        lbph.train(faces, labels)
        print("Trained on 2 sample images")

        # Test prediction
        label, confidence = lbph.predict(gray)
        print(f"Prediction: Label={label}, Confidence={confidence:.2f}")

    except Exception as e:
        print(f"LBPH demo error: {e}")
else:
    print("Face recognizers require opencv-contrib-python")


# =============================================================================
# 4. FACE LANDMARKS
# =============================================================================
print("\n--- 4. Face Landmarks ---")

landmark_info = """
Face Landmark Detection:

OpenCV provides facemark module for detecting facial landmarks.

Available models:
  1. FacemarkLBF - Local Binary Features
  2. FacemarkKazemi - Kazemi's model
  3. FacemarkAAM - Active Appearance Model

Usage:
  facemark = cv2.face.createFacemarkLBF()
  facemark.loadModel('lbfmodel.yaml')

  # Detect faces first
  faces = face_cascade.detectMultiScale(gray)

  # Get landmarks
  ok, landmarks = facemark.fit(gray, faces)

  # landmarks[i] contains 68 points for face i

68 Landmark Points:
  0-16: Jaw line
  17-21: Left eyebrow
  22-26: Right eyebrow
  27-35: Nose
  36-41: Left eye
  42-47: Right eye
  48-67: Mouth
"""
print(landmark_info)


# =============================================================================
# 5. FACE ALIGNMENT
# =============================================================================
print("\n--- 5. Face Alignment ---")


def align_face(img, left_eye, right_eye, desired_size=(256, 256)):
    """Align face based on eye positions."""
    # Calculate angle
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Calculate center
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)

    # Rotate
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return rotated


# Demo alignment
left_eye = (110, 120)
right_eye = (190, 120)
aligned = align_face(face_img, left_eye, right_eye)
print("Face alignment: Rotate based on eye positions")


# =============================================================================
# 6. COMPARISON WITH MODERN APPROACHES
# =============================================================================
print("\n--- 6. Modern Face Recognition ---")

modern_info = """
Modern Face Recognition Libraries:

1. face_recognition (dlib-based):
   pip install face_recognition
   - Simple API
   - High accuracy (99.38% on LFW)
   - Returns face encodings (128-d vectors)

2. DeepFace:
   pip install deepface
   - Multiple backend models
   - VGG-Face, Facenet, OpenFace, etc.
   - Age, gender, emotion detection

3. InsightFace:
   pip install insightface
   - State-of-the-art accuracy
   - ArcFace model
   - GPU recommended

4. MediaPipe Face:
   pip install mediapipe
   - Real-time performance
   - 468 face landmarks
   - Face mesh for AR

Comparison with OpenCV face module:
  - OpenCV: Good for learning, limited accuracy
  - Modern libs: Production-ready, higher accuracy
"""
print(modern_info)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display face module demos."""

    # Show synthetic face
    cv2.imshow("Synthetic Face", face_img)

    # Draw eye positions
    display = face_img.copy()
    cv2.circle(display, (110, 120), 5, (0, 255, 0), -1)
    cv2.circle(display, (190, 120), 5, (0, 255, 0), -1)
    cv2.line(display, (110, 120), (190, 120), (0, 255, 255), 2)
    cv2.putText(display, "Eye alignment", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Face Alignment Demo", display)

    # Show aligned
    cv2.imshow("Aligned Face", aligned)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running face module demonstrations...")
    print("=" * 60)
    show_demo()
