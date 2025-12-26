"""
Application 08: Lane Detection
==============================
Detect road lanes for autonomous driving applications.

Techniques Used:
- Canny edge detection
- Region of interest masking
- Hough line transform
- Line filtering and averaging

Official Docs:
- https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image, get_video


class LaneDetector:
    """
    Lane detection using classical computer vision techniques.
    """

    def __init__(self):
        # Canny thresholds
        self.canny_low = 50
        self.canny_high = 150

        # Hough transform parameters
        self.rho = 2
        self.theta = np.pi / 180
        self.threshold = 50
        self.min_line_length = 40
        self.max_line_gap = 100

    def grayscale(self, image):
        """Convert to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, image, kernel_size=5):
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def canny_edges(self, image):
        """Apply Canny edge detection."""
        return cv2.Canny(image, self.canny_low, self.canny_high)

    def region_of_interest(self, image):
        """
        Apply region of interest mask.
        Focus on lower portion of image where lanes are.
        """
        height, width = image.shape[:2]

        # Define polygon for region of interest
        # Trapezoidal shape covering road area
        polygon = np.array([
            [
                (int(width * 0.1), height),           # Bottom left
                (int(width * 0.45), int(height * 0.6)),  # Top left
                (int(width * 0.55), int(height * 0.6)),  # Top right
                (int(width * 0.9), height)            # Bottom right
            ]
        ], dtype=np.int32)

        # Create mask
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygon, 255)

        # Apply mask
        masked = cv2.bitwise_and(image, mask)

        return masked, polygon

    def detect_lines(self, image):
        """
        Detect lines using Hough transform.
        """
        lines = cv2.HoughLinesP(
            image,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        return lines

    def average_slope_intercept(self, image, lines):
        """
        Separate lines into left and right lanes based on slope.
        Average the lines to get single left and right lane lines.
        """
        left_fit = []
        right_fit = []

        if lines is None:
            return None, None

        height, width = image.shape[:2]

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            # Calculate slope and intercept
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Filter by slope (ignore near-horizontal lines)
            if abs(slope) < 0.5:
                continue

            # Left lane has negative slope, right has positive
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        # Average the lines
        left_line = None
        right_line = None

        if left_fit:
            left_avg = np.average(left_fit, axis=0)
            left_line = self.make_coordinates(image, left_avg)

        if right_fit:
            right_avg = np.average(right_fit, axis=0)
            right_line = self.make_coordinates(image, right_avg)

        return left_line, right_line

    def make_coordinates(self, image, line_params):
        """
        Convert slope/intercept to line coordinates.
        """
        slope, intercept = line_params
        height = image.shape[0]

        # Line from bottom to 60% of image height
        y1 = height
        y2 = int(height * 0.6)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])

    def draw_lines(self, image, left_line, right_line, color=(0, 255, 0), thickness=10):
        """
        Draw lane lines on image.
        """
        line_image = np.zeros_like(image)

        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

        # Fill lane area
        if left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)
            cv2.fillPoly(line_image, [pts], (0, 100, 0))

        return line_image

    def process_frame(self, frame):
        """
        Full lane detection pipeline.
        """
        # Convert to grayscale
        gray = self.grayscale(frame)

        # Apply blur
        blur = self.gaussian_blur(gray)

        # Edge detection
        edges = self.canny_edges(blur)

        # Region of interest
        masked, roi_polygon = self.region_of_interest(edges)

        # Detect lines
        lines = self.detect_lines(masked)

        # Average lines
        left_line, right_line = self.average_slope_intercept(frame, lines)

        # Draw lines
        line_image = self.draw_lines(frame, left_line, right_line)

        # Blend with original
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

        # Draw ROI boundary
        cv2.polylines(result, [roi_polygon], True, (255, 255, 0), 2)

        return result, edges, masked


def create_road_image():
    """
    Create a synthetic road image for demo.
    """
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Sky (gradient)
    for y in range(200):
        blue = 200 - y // 2
        img[y, :] = (blue + 55, blue + 30, blue)

    # Road (gray)
    road_pts = np.array([
        [0, 400], [200, 200], [400, 200], [600, 400]
    ], dtype=np.int32)
    cv2.fillPoly(img, [road_pts], (80, 80, 80))

    # Lane markings (white dashed lines)
    # Left lane
    for i in range(5):
        y1 = 400 - i * 40
        y2 = y1 - 20
        x1 = int(150 + i * 15)
        x2 = int(150 + (i + 0.5) * 15)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

    # Right lane
    for i in range(5):
        y1 = 400 - i * 40
        y2 = y1 - 20
        x1 = int(450 - i * 15)
        x2 = int(450 - (i + 0.5) * 15)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

    # Center line (yellow)
    for i in range(5):
        y1 = 400 - i * 40
        y2 = y1 - 20
        cv2.line(img, (300, y1), (300, y2), (0, 200, 255), 2)

    return img


def load_road_image():
    """
    Load a real road image or create synthetic one.
    """
    # Try to load road sample
    for sample in ["road.jpg", "highway.jpg", "lane.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Create synthetic road
    print("No road sample found. Using synthetic road image.")
    return create_road_image()


def interactive_lane_detection():
    """
    Interactive lane detection with webcam or video.
    """
    # Try video file first
    video_path = get_video("road.mp4")
    cap = None

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"Using video: road.mp4")

    # Fall back to webcam
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera. Using demo mode.")
            demo_mode()
            return
        print("Using webcam (point at road markings)")

    print("\n=== Lane Detection ===")
    print("Controls:")
    print("  '+'/'-' - Adjust Canny thresholds")
    print("  's' - Save screenshot")
    print("  'q' - Quit")
    print("======================\n")

    detector = LaneDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Process frame
        result, edges, masked = detector.process_frame(frame)

        # Display info
        cv2.putText(result, f"Canny: {detector.canny_low}-{detector.canny_high}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Lane Detection", result)
        cv2.imshow("Edges", edges)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            detector.canny_high = min(300, detector.canny_high + 10)
            detector.canny_low = min(detector.canny_high - 50, detector.canny_low + 5)
        elif key == ord('-'):
            detector.canny_low = max(10, detector.canny_low - 5)
            detector.canny_high = max(detector.canny_low + 50, detector.canny_high - 10)
        elif key == ord('s'):
            cv2.imwrite("lane_detection.jpg", result)
            print("Saved: lane_detection.jpg")

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with static road image.
    """
    print("\n=== Lane Detection Demo ===\n")

    # Load road image
    img = load_road_image()

    detector = LaneDetector()

    # Process image
    result, edges, masked = detector.process_frame(img)

    # Show steps
    gray = detector.grayscale(img)
    blur = detector.gaussian_blur(gray)

    # Create visualization
    steps = np.hstack([
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    ])
    steps = cv2.resize(steps, (900, 200))

    cv2.putText(steps, "Grayscale", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(steps, "Canny Edges", (310, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(steps, "ROI Masked", (610, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Processing Steps", steps)
    cv2.imshow("Lane Detection Result", result)

    print("Lane detection pipeline:")
    print("1. Convert to grayscale")
    print("2. Apply Gaussian blur")
    print("3. Canny edge detection")
    print("4. Mask region of interest")
    print("5. Hough line detection")
    print("6. Average and draw lanes")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 08: Lane Detection")
    print("=" * 60)

    try:
        interactive_lane_detection()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
