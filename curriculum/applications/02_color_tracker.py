"""
Application 02: Color Object Tracker
=====================================
Track colored objects in real-time using HSV color space.

Techniques Used:
- Color space conversion (BGR to HSV)
- Color thresholding (inRange)
- Morphological operations
- Contour detection
- Bounding box tracking

Official Docs:
- https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


# Predefined color ranges in HSV
COLOR_RANGES = {
    'red': {
        'lower1': np.array([0, 100, 100]),
        'upper1': np.array([10, 255, 255]),
        'lower2': np.array([160, 100, 100]),  # Red wraps around
        'upper2': np.array([180, 255, 255]),
    },
    'green': {
        'lower': np.array([35, 100, 100]),
        'upper': np.array([85, 255, 255]),
    },
    'blue': {
        'lower': np.array([100, 100, 100]),
        'upper': np.array([130, 255, 255]),
    },
    'yellow': {
        'lower': np.array([20, 100, 100]),
        'upper': np.array([35, 255, 255]),
    },
    'orange': {
        'lower': np.array([10, 100, 100]),
        'upper': np.array([20, 255, 255]),
    },
    'purple': {
        'lower': np.array([130, 100, 100]),
        'upper': np.array([160, 255, 255]),
    },
}


def create_color_mask(hsv_image, color_name):
    """
    Create a binary mask for the specified color.
    """
    if color_name not in COLOR_RANGES:
        return None

    color = COLOR_RANGES[color_name]

    if 'lower2' in color:  # Handle red (wraps around hue)
        mask1 = cv2.inRange(hsv_image, color['lower1'], color['upper1'])
        mask2 = cv2.inRange(hsv_image, color['lower2'], color['upper2'])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv_image, color['lower'], color['upper'])

    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


def find_largest_contour(mask, min_area=500):
    """
    Find the largest contour in the mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < min_area:
        return None

    return largest


def track_object(frame, color_name):
    """
    Track an object of specified color and return its position.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = create_color_mask(hsv, color_name)

    if mask is None:
        return frame, None, None

    # Find largest contour
    contour = find_largest_contour(mask)

    center = None
    bbox = None

    if contour is not None:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, w, h)

        # Get center using moments
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if center:
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{color_name.upper()}: ({cx}, {cy})",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, center, mask


class ColorTracker:
    """
    Multi-color object tracker with trail visualization.
    """

    def __init__(self, max_trail_length=50):
        self.trails = {}
        self.max_trail_length = max_trail_length

    def update(self, frame, colors):
        """
        Track multiple colors and draw trails.
        """
        result = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name in colors:
            mask = create_color_mask(hsv, color_name)
            if mask is None:
                continue

            contour = find_largest_contour(mask)

            if contour is not None:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Update trail
                    if color_name not in self.trails:
                        self.trails[color_name] = []
                    self.trails[color_name].append((cx, cy))

                    # Limit trail length
                    if len(self.trails[color_name]) > self.max_trail_length:
                        self.trails[color_name].pop(0)

                    # Draw bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)

            # Draw trail
            if color_name in self.trails and len(self.trails[color_name]) > 1:
                points = np.array(self.trails[color_name], dtype=np.int32)
                cv2.polylines(result, [points], False, (255, 0, 255), 2)

        return result

    def clear_trails(self):
        """Clear all trails."""
        self.trails = {}


def create_hsv_trackbars(window_name="HSV Tuner"):
    """
    Create trackbars for HSV tuning.
    """
    cv2.namedWindow(window_name)

    cv2.createTrackbar("H Min", window_name, 0, 180, lambda x: None)
    cv2.createTrackbar("H Max", window_name, 180, 180, lambda x: None)
    cv2.createTrackbar("S Min", window_name, 100, 255, lambda x: None)
    cv2.createTrackbar("S Max", window_name, 255, 255, lambda x: None)
    cv2.createTrackbar("V Min", window_name, 100, 255, lambda x: None)
    cv2.createTrackbar("V Max", window_name, 255, 255, lambda x: None)


def get_hsv_values(window_name="HSV Tuner"):
    """
    Get current trackbar values.
    """
    h_min = cv2.getTrackbarPos("H Min", window_name)
    h_max = cv2.getTrackbarPos("H Max", window_name)
    s_min = cv2.getTrackbarPos("S Min", window_name)
    s_max = cv2.getTrackbarPos("S Max", window_name)
    v_min = cv2.getTrackbarPos("V Min", window_name)
    v_max = cv2.getTrackbarPos("V Max", window_name)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    return lower, upper


def hsv_tuner_mode():
    """
    Interactive HSV tuner to find color ranges.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("\n=== HSV Color Tuner ===")
    print("Adjust sliders to find your color range")
    print("Press 'q' to quit")
    print("=======================\n")

    create_hsv_trackbars()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = get_hsv_values()

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Show HSV values
        cv2.putText(frame, f"Lower: {lower}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Upper: {upper}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display = np.hstack([frame, result])
        cv2.imshow("HSV Tuner", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nYour color range:")
    print(f"Lower: np.array({list(lower)})")
    print(f"Upper: np.array({list(upper)})")


def interactive_tracker():
    """
    Interactive color tracker with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== Color Object Tracker ===")
    print("Controls:")
    print("  'r' - Track RED")
    print("  'g' - Track GREEN")
    print("  'b' - Track BLUE")
    print("  'y' - Track YELLOW")
    print("  'o' - Track ORANGE")
    print("  'p' - Track PURPLE")
    print("  't' - HSV Tuner mode")
    print("  'c' - Clear trails")
    print("  'q' - Quit")
    print("============================\n")

    current_color = 'blue'
    tracker = ColorTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track object
        result = tracker.update(frame, [current_color])

        # Display info
        cv2.putText(result, f"Tracking: {current_color.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Color Tracker", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            current_color = 'red'
        elif key == ord('g'):
            current_color = 'green'
        elif key == ord('b'):
            current_color = 'blue'
        elif key == ord('y'):
            current_color = 'yellow'
        elif key == ord('o'):
            current_color = 'orange'
        elif key == ord('p'):
            current_color = 'purple'
        elif key == ord('c'):
            tracker.clear_trails()
        elif key == ord('t'):
            cap.release()
            cv2.destroyAllWindows()
            hsv_tuner_mode()
            cap = cv2.VideoCapture(0)

    cap.release()
    cv2.destroyAllWindows()


def load_demo_image():
    """
    Load a real colorful image for demo, or create one if not available.
    """
    # Try to load colorful sample images
    for sample in ["fruits.jpg", "lena.jpg", "baboon.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Fallback: create demo image with colored objects
    print("No color sample found. Using synthetic demo.")
    print("Run: python curriculum/sample_data/download_samples.py")

    img = np.ones((400, 600, 3), dtype=np.uint8) * 240

    # Draw colored circles
    cv2.circle(img, (100, 200), 50, (0, 0, 255), -1)    # Red
    cv2.circle(img, (250, 200), 50, (0, 255, 0), -1)    # Green
    cv2.circle(img, (400, 200), 50, (255, 0, 0), -1)    # Blue
    cv2.circle(img, (500, 100), 40, (0, 255, 255), -1)  # Yellow

    return img


def demo_mode():
    """
    Demo with real colorful image or colored circles.
    """
    print("\n=== Color Tracker Demo ===\n")

    # Load real image or create demo
    img = load_demo_image()

    # Track each color
    colors = ['red', 'green', 'blue', 'yellow']
    result = img.copy()

    for color in colors:
        result, center, mask = track_object(result, color)
        if center:
            print(f"{color.upper()} detected at {center}")

    cv2.imshow("Color Tracking Demo", result)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 02: Color Object Tracker")
    print("=" * 60)

    try:
        interactive_tracker()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
