"""
Module 5: Object Detection - Cascade Classifiers
=================================================
Using Haar and LBP cascade classifiers for object detection.

Official Docs: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html

Topics Covered:
1. Loading Cascade Classifiers
2. Face Detection
3. Eye Detection
4. Detection Parameters
5. Multiple Object Detection
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Module 5: Cascade Classifiers")
print("=" * 60)


# =============================================================================
# 1. LOADING CASCADE CLASSIFIERS
# =============================================================================
print("\n--- 1. Loading Cascade Classifiers ---")

# OpenCV comes with pre-trained cascades
# Location varies by installation

# Common cascade files:
cascades_info = """
Pre-trained Haar Cascades:
  haarcascade_frontalface_default.xml  - Frontal face
  haarcascade_frontalface_alt.xml      - Alternative frontal face
  haarcascade_frontalface_alt2.xml     - Another alternative
  haarcascade_profileface.xml          - Side profile face
  haarcascade_eye.xml                  - Eyes
  haarcascade_eye_tree_eyeglasses.xml  - Eyes with glasses
  haarcascade_smile.xml                - Smile
  haarcascade_upperbody.xml            - Upper body
  haarcascade_lowerbody.xml            - Lower body
  haarcascade_fullbody.xml             - Full body
  haarcascade_frontalcatface.xml       - Cat face
"""
print(cascades_info)

# Load face cascade
# Try to find cascade file
cascade_paths = [
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
]

face_cascade = None
for path in cascade_paths:
    if os.path.exists(path):
        face_cascade = cv2.CascadeClassifier(path)
        print(f"Loaded face cascade from: {path}")
        break

if face_cascade is None or face_cascade.empty():
    # Create using OpenCV's built-in path
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not face_cascade.empty():
        print("Loaded face cascade from cv2.data.haarcascades")

# Load eye cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# =============================================================================
# 2. LOAD TEST IMAGE
# =============================================================================
def load_face_image():
    """Load image with faces for detection demo."""
    # Try sample images with faces
    for sample in ["lena.jpg", "lena_face.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Try local file
    if os.path.exists("face.png"):
        img = cv2.imread("face.png")
        if img is not None:
            print("Using local image: face.png")
            return img

    # Try webcam capture
    print("No face image found. Trying webcam...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("Using webcam capture")
            return frame

    # Final fallback: create placeholder
    print("No image source available.")
    print("Run: python curriculum/sample_data/download_samples.py")
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(img, "No face image", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


test_img = load_face_image()
gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)


# =============================================================================
# 3. DETECTION PARAMETERS
# =============================================================================
print("\n--- 3. Detection Parameters ---")

params_info = """
detectMultiScale Parameters:

  scaleFactor (1.01-1.5):
    - How much to reduce image size at each scale
    - Lower = more accurate but slower
    - Typical: 1.1-1.3

  minNeighbors (0-10):
    - How many neighbors each rectangle should have
    - Higher = fewer false positives, might miss faces
    - Typical: 3-6

  minSize (width, height):
    - Minimum object size
    - Skip smaller detections
    - (30, 30) is common for faces

  maxSize (width, height):
    - Maximum object size
    - Skip larger detections
"""
print(params_info)


# =============================================================================
# 4. FACE DETECTION
# =============================================================================
print("\n--- 4. Face Detection ---")

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Faces detected: {len(faces)}")

# Draw rectangles around faces
face_img = test_img.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print(f"  Face at ({x}, {y}) - size: {w}x{h}")


# =============================================================================
# 5. EYE DETECTION
# =============================================================================
print("\n--- 5. Eye Detection ---")

# Detect eyes (within face regions for better accuracy)
eye_img = test_img.copy()
all_eyes = []

for (x, y, w, h) in faces:
    # Region of interest - face area
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = eye_img[y:y+h, x:x+w]

    # Detect eyes in face region
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        # Draw on the ROI
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        all_eyes.append((x+ex, y+ey, ew, eh))

print(f"Eyes detected: {len(all_eyes)}")


# =============================================================================
# 6. DETECTION WITH DIFFERENT PARAMETERS
# =============================================================================
print("\n--- 6. Parameter Comparison ---")


def detect_with_params(img, cascade, scale, neighbors, min_size):
    """Detect objects with specific parameters."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(
        gray_img,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=min_size
    )
    return detections


# Test different parameters
configs = [
    (1.05, 3, (20, 20)),   # Sensitive
    (1.1, 5, (30, 30)),    # Balanced
    (1.3, 8, (50, 50)),    # Strict
]

print("\nParameter comparison:")
for scale, neighbors, min_size in configs:
    detections = detect_with_params(test_img, face_cascade, scale, neighbors, min_size)
    print(f"  scale={scale}, neighbors={neighbors}, minSize={min_size}: {len(detections)} faces")


# =============================================================================
# 7. PERFORMANCE TIPS
# =============================================================================
print("\n--- 7. Performance Tips ---")

tips = """
Cascade Classifier Tips:

1. Image Size:
   - Smaller images = faster detection
   - Resize before detection, scale coordinates back

2. ROI Detection:
   - Detect in regions of interest only
   - Eyes within faces, faces within upper body

3. Skip Frames:
   - For video, detect every N frames
   - Track between detections

4. Threading:
   - detectMultiScale can be slow
   - Consider background threads

5. Alternative:
   - For better accuracy, use DNN-based detection
   - cv2.dnn with face detection models
"""
print(tips)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display cascade classifier demos."""

    # Original
    cv2.imshow("Original", test_img)

    # Face detection
    cv2.putText(face_img, f"Faces: {len(faces)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Face Detection", face_img)

    # Eye detection
    cv2.putText(eye_img, f"Eyes: {len(all_eyes)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Eye Detection", eye_img)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running cascade classifier demonstrations...")
    print("=" * 60)
    show_demo()
