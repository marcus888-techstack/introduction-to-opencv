"""
Application 04: Face Blur Privacy Filter
=========================================
Automatically detect and blur faces for privacy protection.

Techniques Used:
- Haar cascade face detection
- Gaussian blur
- Pixelation
- ROI manipulation

Official Docs:
- https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class FaceBlurrer:
    """
    Face detection and blurring for privacy protection.
    """

    def __init__(self):
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Detection parameters
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)

    def detect_faces(self, image):
        """
        Detect faces in image.
        Returns list of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        return faces

    def blur_face(self, image, x, y, w, h, blur_type='gaussian', intensity=51):
        """
        Blur a specific face region.

        blur_type: 'gaussian', 'pixelate', 'solid'
        """
        result = image.copy()

        # Extract face ROI
        face_roi = result[y:y+h, x:x+w]

        if blur_type == 'gaussian':
            # Gaussian blur
            blurred = cv2.GaussianBlur(face_roi, (intensity, intensity), 0)

        elif blur_type == 'pixelate':
            # Pixelate
            pixel_size = max(1, intensity // 5)
            small = cv2.resize(face_roi,
                              (max(1, w // pixel_size), max(1, h // pixel_size)),
                              interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif blur_type == 'solid':
            # Solid color block
            blurred = np.full_like(face_roi, (128, 128, 128))

        elif blur_type == 'emoji':
            # Simple emoji-like circle
            blurred = face_roi.copy()
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            cv2.circle(blurred, center, radius, (0, 200, 255), -1)
            # Eyes
            cv2.circle(blurred, (w // 3, h // 3), radius // 6, (0, 0, 0), -1)
            cv2.circle(blurred, (2 * w // 3, h // 3), radius // 6, (0, 0, 0), -1)
            # Smile
            cv2.ellipse(blurred, (w // 2, int(h * 0.6)), (radius // 2, radius // 3),
                       0, 0, 180, (0, 0, 0), 2)

        else:
            blurred = face_roi

        # Apply to result
        result[y:y+h, x:x+w] = blurred

        return result

    def blur_ellipse(self, image, x, y, w, h, blur_type='gaussian', intensity=51):
        """
        Blur face with elliptical mask (more natural looking).
        """
        result = image.copy()

        # Create elliptical mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Create blurred version
        face_roi = result[y:y+h, x:x+w].copy()

        if blur_type == 'gaussian':
            blurred = cv2.GaussianBlur(face_roi, (intensity, intensity), 0)
        elif blur_type == 'pixelate':
            pixel_size = max(1, intensity // 5)
            small = cv2.resize(face_roi,
                              (max(1, w // pixel_size), max(1, h // pixel_size)),
                              interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            blurred = face_roi

        # Apply mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        face_roi = face_roi.astype(float)
        blurred = blurred.astype(float)

        combined = face_roi * (1 - mask_3ch) + blurred * mask_3ch
        result[y:y+h, x:x+w] = combined.astype(np.uint8)

        return result

    def process_image(self, image, blur_type='gaussian', intensity=51, ellipse=True):
        """
        Detect and blur all faces in image.
        """
        faces = self.detect_faces(image)
        result = image.copy()

        for (x, y, w, h) in faces:
            # Add padding
            pad = int(w * 0.1)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2 * pad)
            h = min(image.shape[0] - y, h + 2 * pad)

            if ellipse:
                result = self.blur_ellipse(result, x, y, w, h, blur_type, intensity)
            else:
                result = self.blur_face(result, x, y, w, h, blur_type, intensity)

        return result, len(faces)


def load_face_image():
    """
    Load a real face image for demo, or create one if not available.
    """
    # Try to load face sample images
    for sample in ["lena.jpg", "lena_face.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
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

    # Fallback: create test image with face-like features
    print("No image source available. Using synthetic faces.")
    print("Run: python curriculum/sample_data/download_samples.py")
    return create_test_image()


def create_test_image():
    """
    Create a test image with face-like features (fallback).
    """
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200

    # Background texture
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    # Draw simplified faces
    faces = [(150, 200, 80), (350, 180, 70), (500, 220, 60)]

    for x, y, r in faces:
        # Face oval
        cv2.ellipse(img, (x, y), (r, int(r * 1.3)), 0, 0, 360, (180, 160, 140), -1)
        # Eyes
        cv2.circle(img, (x - r // 3, y - r // 4), r // 6, (50, 50, 50), -1)
        cv2.circle(img, (x + r // 3, y - r // 4), r // 6, (50, 50, 50), -1)
        # Nose
        cv2.line(img, (x, y - r // 8), (x, y + r // 4), (120, 100, 90), 2)
        # Mouth
        cv2.ellipse(img, (x, y + r // 2), (r // 3, r // 6), 0, 0, 180, (150, 100, 100), 2)

    return img


def interactive_blur():
    """
    Interactive face blur with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== Face Blur Privacy Filter ===")
    print("Controls:")
    print("  '1' - Gaussian blur")
    print("  '2' - Pixelate")
    print("  '3' - Solid block")
    print("  '4' - Emoji cover")
    print("  'e' - Toggle ellipse mask")
    print("  '+' - Increase blur intensity")
    print("  '-' - Decrease blur intensity")
    print("  's' - Save screenshot")
    print("  'q' - Quit")
    print("================================\n")

    blurrer = FaceBlurrer()
    blur_type = 'gaussian'
    intensity = 51
    use_ellipse = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result, face_count = blurrer.process_image(
            frame, blur_type, intensity, use_ellipse
        )

        # Display info
        info = f"Blur: {blur_type} | Intensity: {intensity} | Faces: {face_count}"
        cv2.putText(result, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Face Blur", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            blur_type = 'gaussian'
        elif key == ord('2'):
            blur_type = 'pixelate'
        elif key == ord('3'):
            blur_type = 'solid'
        elif key == ord('4'):
            blur_type = 'emoji'
        elif key == ord('e'):
            use_ellipse = not use_ellipse
        elif key == ord('+') or key == ord('='):
            intensity = min(101, intensity + 10)
            if intensity % 2 == 0:
                intensity += 1
        elif key == ord('-'):
            intensity = max(11, intensity - 10)
            if intensity % 2 == 0:
                intensity += 1
        elif key == ord('s'):
            cv2.imwrite("blurred_output.jpg", result)
            print("Saved: blurred_output.jpg")

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with real face image or test image.
    """
    print("\n=== Face Blur Demo ===\n")

    # Load a real face image or create test
    img = load_face_image()

    blurrer = FaceBlurrer()

    # Show different blur types
    blur_types = ['gaussian', 'pixelate', 'solid', 'emoji']

    results = [img]
    for bt in blur_types:
        result, count = blurrer.process_image(img, bt, intensity=51, ellipse=False)
        cv2.putText(result, bt.upper(), (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        results.append(result)

    # Stack results
    row1 = np.hstack(results[:3])
    row2 = np.hstack(results[3:] + [np.zeros_like(img)])
    display = np.vstack([row1, row2])
    display = cv2.resize(display, (900, 600))

    cv2.imshow("Blur Types", display)
    print("Showing different blur types...")
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blur_image_file(input_path, output_path, blur_type='gaussian'):
    """
    Blur faces in an image file.
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"Cannot read image: {input_path}")
        return False

    blurrer = FaceBlurrer()
    result, count = blurrer.process_image(img, blur_type)

    cv2.imwrite(output_path, result)
    print(f"Processed {count} faces. Saved to: {output_path}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Application 04: Face Blur Privacy Filter")
    print("=" * 60)

    try:
        interactive_blur()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
