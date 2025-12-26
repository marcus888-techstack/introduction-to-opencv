"""
Application 12: Hand Gesture Recognition
========================================
Detect and recognize hand gestures using contour analysis.

Techniques Used:
- Skin color segmentation (HSV)
- Contour detection
- Convex hull and convexity defects
- Finger counting

Official Docs:
- https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class HandGestureRecognizer:
    """
    Hand gesture recognition using classical computer vision.
    """

    def __init__(self):
        # HSV ranges for skin detection (adjustable)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Additional skin range (handles different lighting)
        self.lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        self.upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

        # Background subtractor for motion-based detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

        # Gesture names
        self.gestures = {
            0: "Fist",
            1: "One",
            2: "Two/Peace",
            3: "Three",
            4: "Four",
            5: "Five/Open Hand"
        }

    def detect_skin(self, image):
        """
        Detect skin regions using HSV color space.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for skin color
        mask1 = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Blur to smooth edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        return mask

    def find_hand_contour(self, mask):
        """
        Find the largest contour (assumed to be hand).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # Filter by size
        if cv2.contourArea(max_contour) < 5000:
            return None

        return max_contour

    def count_fingers(self, contour, image_shape):
        """
        Count fingers using convexity defects.
        Returns: (finger_count, hull, defects)
        """
        if contour is None:
            return 0, None, None

        # Get convex hull
        hull = cv2.convexHull(contour)

        # Get convex hull indices for defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)

        # Calculate convexity defects
        if len(hull_indices) < 4:
            return 0, hull, None

        defects = cv2.convexityDefects(contour, hull_indices)

        if defects is None:
            return 0, hull, None

        # Count fingers by analyzing defects
        finger_count = 0
        h, w = image_shape[:2]

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Calculate angle between fingers
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # Cosine rule
            if b * c == 0:
                continue

            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
            angle_deg = np.degrees(angle)

            # If angle is less than 90 degrees and defect is significant
            if angle_deg <= 90 and d > 10000:
                finger_count += 1

        # Add 1 for the thumb (if fingers detected)
        if finger_count > 0:
            finger_count += 1

        # Clamp to 0-5
        finger_count = min(5, max(0, finger_count))

        return finger_count, hull, defects

    def get_gesture_name(self, finger_count):
        """Get gesture name from finger count."""
        return self.gestures.get(finger_count, "Unknown")

    def draw_hand_analysis(self, image, contour, hull, defects, finger_count):
        """
        Draw hand analysis visualization.
        """
        result = image.copy()

        if contour is None:
            return result

        # Draw contour
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

        # Draw hull
        if hull is not None:
            cv2.drawContours(result, [hull], 0, (0, 0, 255), 2)

        # Draw defects
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]

                if d > 10000:  # Significant defects only
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    cv2.circle(result, start, 8, (255, 0, 0), -1)
                    cv2.circle(result, end, 8, (255, 0, 0), -1)
                    cv2.circle(result, far, 8, (0, 255, 255), -1)
                    cv2.line(result, start, far, (255, 255, 0), 2)
                    cv2.line(result, end, far, (255, 255, 0), 2)

        # Draw center and bounding box
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 10, (255, 0, 255), -1)

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Display gesture
        gesture = self.get_gesture_name(finger_count)
        cv2.putText(result, f"Fingers: {finger_count} - {gesture}",
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return result

    def process_frame(self, frame):
        """
        Full hand gesture recognition pipeline.
        """
        # Detect skin
        mask = self.detect_skin(frame)

        # Find hand contour
        contour = self.find_hand_contour(mask)

        # Count fingers
        finger_count, hull, defects = self.count_fingers(contour, frame.shape)

        # Draw analysis
        result = self.draw_hand_analysis(frame, contour, hull, defects, finger_count)

        return result, mask, finger_count


def load_hand_image():
    """
    Load an image with a hand or create synthetic one.
    """
    # Try sample images
    for sample in ["hand.jpg", "hand_gesture.jpg", "palm.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Create synthetic hand
    print("No hand sample found. Using synthetic hand.")

    img = np.ones((400, 500, 3), dtype=np.uint8) * 100

    # Draw simplified hand shape (skin color)
    skin_color = (140, 180, 230)  # BGR approximating skin

    # Palm
    cv2.ellipse(img, (250, 280), (80, 100), 0, 0, 360, skin_color, -1)

    # Fingers
    finger_tips = [
        ((170, 120), (190, 200)),  # Thumb
        ((200, 80), (210, 180)),   # Index
        ((250, 60), (255, 180)),   # Middle
        ((300, 80), (295, 180)),   # Ring
        ((340, 120), (320, 200)),  # Pinky
    ]

    for tip, base in finger_tips:
        cv2.line(img, base, tip, skin_color, 30)
        cv2.circle(img, tip, 15, skin_color, -1)

    cv2.putText(img, "Synthetic Hand", (150, 380),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return img


def interactive_gesture():
    """
    Interactive hand gesture recognition with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== Hand Gesture Recognition ===")
    print("Controls:")
    print("  'h' - Adjust HSV lower bound")
    print("  'H' - Adjust HSV upper bound")
    print("  's' - Save screenshot")
    print("  'r' - Reset skin detection")
    print("  'q' - Quit")
    print("================================\n")
    print("Show your hand to the camera!")

    recognizer = HandGestureRecognizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Process frame
        result, mask, finger_count = recognizer.process_frame(frame)

        # Display info
        cv2.putText(result, f"Detected Fingers: {finger_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        gesture = recognizer.get_gesture_name(finger_count)
        cv2.putText(result, f"Gesture: {gesture}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show windows
        cv2.imshow("Hand Gesture Recognition", result)
        cv2.imshow("Skin Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('h'):
            recognizer.lower_skin[0] = (recognizer.lower_skin[0] + 5) % 180
            print(f"Lower H: {recognizer.lower_skin[0]}")
        elif key == ord('H'):
            recognizer.upper_skin[0] = (recognizer.upper_skin[0] + 5) % 180
            print(f"Upper H: {recognizer.upper_skin[0]}")
        elif key == ord('r'):
            recognizer.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            recognizer.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            print("Reset skin detection parameters")
        elif key == ord('s'):
            cv2.imwrite("hand_gesture.jpg", result)
            print("Saved: hand_gesture.jpg")

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with static hand image.
    """
    print("\n=== Hand Gesture Demo ===\n")

    # Load hand image
    img = load_hand_image()

    recognizer = HandGestureRecognizer()

    # Process image
    result, mask, finger_count = recognizer.process_frame(img)

    print(f"Detected fingers: {finger_count}")
    print(f"Gesture: {recognizer.get_gesture_name(finger_count)}")

    # Create display
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    display = np.hstack([img, mask_color, result])
    display = cv2.resize(display, (1200, 400))

    cv2.putText(display, "Original", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, "Skin Mask", (410, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display, "Analysis", (810, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Recognition Demo", display)

    print("\nTechniques used:")
    print("- HSV skin color segmentation")
    print("- Contour detection")
    print("- Convex hull analysis")
    print("- Convexity defects for finger counting")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 12: Hand Gesture Recognition")
    print("=" * 60)

    try:
        interactive_gesture()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
