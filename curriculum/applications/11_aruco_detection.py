"""
Application 11: ArUco Marker Detection
======================================
Detect and track ArUco markers for augmented reality applications.

Techniques Used:
- ArUco dictionary generation
- Marker detection and pose estimation
- Camera calibration concepts
- 3D coordinate overlay

Official Docs:
- https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class ArucoDetector:
    """
    ArUco marker detection and pose estimation.
    """

    # Available ArUco dictionaries
    DICTIONARIES = {
        '4x4_50': cv2.aruco.DICT_4X4_50,
        '4x4_100': cv2.aruco.DICT_4X4_100,
        '4x4_250': cv2.aruco.DICT_4X4_250,
        '5x5_50': cv2.aruco.DICT_5X5_50,
        '5x5_100': cv2.aruco.DICT_5X5_100,
        '6x6_50': cv2.aruco.DICT_6X6_50,
        '6x6_100': cv2.aruco.DICT_6X6_100,
        'ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(self, dict_type='4x4_50'):
        # Get dictionary
        dict_id = self.DICTIONARIES.get(dict_type, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Camera matrix (default - should be calibrated for accurate pose)
        self.camera_matrix = None
        self.dist_coeffs = None

    def set_camera_params(self, frame_shape):
        """
        Set default camera parameters based on frame size.
        For accurate pose estimation, use proper camera calibration.
        """
        h, w = frame_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)

        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((4, 1))

    def detect_markers(self, image):
        """
        Detect ArUco markers in image.
        Returns: (corners, ids, rejected)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected

    def draw_markers(self, image, corners, ids):
        """
        Draw detected markers on image.
        """
        result = image.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(result, corners, ids)

            # Add ID labels
            for i, corner in enumerate(corners):
                c = corner[0]
                center = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                cv2.putText(result, f"ID: {ids[i][0]}", (center[0] - 20, center[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return result

    def estimate_pose(self, corners, marker_size=0.05):
        """
        Estimate pose of markers.
        marker_size: Size of marker in meters.
        Returns: (rvecs, tvecs)
        """
        if self.camera_matrix is None:
            return None, None

        rvecs = []
        tvecs = []

        for corner in corners:
            # Define marker corners in 3D
            obj_points = np.array([
                [-marker_size/2, marker_size/2, 0],
                [marker_size/2, marker_size/2, 0],
                [marker_size/2, -marker_size/2, 0],
                [-marker_size/2, -marker_size/2, 0]
            ], dtype=np.float32)

            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                obj_points, corner[0], self.camera_matrix, self.dist_coeffs
            )

            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)

        return rvecs, tvecs

    def draw_axes(self, image, corners, rvecs, tvecs, length=0.03):
        """
        Draw 3D axes on markers.
        """
        result = image.copy()

        if rvecs is None or len(rvecs) == 0:
            return result

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(result, self.camera_matrix, self.dist_coeffs,
                             rvec, tvec, length)

        return result

    def get_marker_center(self, corner):
        """Get center point of marker."""
        c = corner[0]
        return (int(c[:, 0].mean()), int(c[:, 1].mean()))

    def get_marker_area(self, corner):
        """Get area of marker."""
        c = corner[0]
        return cv2.contourArea(c.astype(np.float32))


def generate_marker(dict_type='4x4_50', marker_id=0, size=200):
    """
    Generate an ArUco marker image.
    """
    dict_id = ArucoDetector.DICTIONARIES.get(dict_type, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)

    return marker


def generate_marker_board(dict_type='4x4_50', rows=2, cols=3, marker_size=100, margin=20):
    """
    Generate a board with multiple markers.
    """
    board_width = cols * marker_size + (cols + 1) * margin
    board_height = rows * marker_size + (rows + 1) * margin

    board = np.ones((board_height, board_width), dtype=np.uint8) * 255

    marker_id = 0
    for r in range(rows):
        for c in range(cols):
            marker = generate_marker(dict_type, marker_id, marker_size)
            x = margin + c * (marker_size + margin)
            y = margin + r * (marker_size + margin)
            board[y:y+marker_size, x:x+marker_size] = marker
            marker_id += 1

    return board


def load_aruco_image():
    """
    Load an image with ArUco markers or create one.
    """
    # Try sample images
    for sample in ["aruco.png", "aruco_markers.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Generate image with markers
    print("No ArUco sample found. Generating markers.")

    # Create background
    img = np.ones((500, 700, 3), dtype=np.uint8) * 240

    # Add multiple markers at different positions and scales
    markers = [
        (generate_marker('4x4_50', 0, 100), (50, 50)),
        (generate_marker('4x4_50', 1, 80), (500, 50)),
        (generate_marker('4x4_50', 2, 120), (100, 300)),
        (generate_marker('4x4_50', 3, 90), (450, 320)),
    ]

    for marker, (x, y) in markers:
        h, w = marker.shape
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img[y:y+h, x:x+w] = marker_bgr

    cv2.putText(img, "Generated ArUco Markers (4x4_50)", (150, 480),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    return img


def interactive_detector():
    """
    Interactive ArUco marker detection with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== ArUco Marker Detection ===")
    print("Controls:")
    print("  '1'-'4' - Change dictionary")
    print("  'p' - Toggle pose estimation")
    print("  'g' - Generate and save marker")
    print("  's' - Save screenshot")
    print("  'q' - Quit")
    print("==============================\n")

    detector = ArucoDetector('4x4_50')
    show_pose = False
    dict_names = ['4x4_50', '5x5_50', '6x6_50', 'ARUCO_ORIGINAL']
    current_dict = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Set camera params if not set
        if detector.camera_matrix is None:
            detector.set_camera_params(frame.shape)

        # Detect markers
        corners, ids, rejected = detector.detect_markers(frame)

        # Draw markers
        result = detector.draw_markers(frame, corners, ids)

        # Pose estimation
        if show_pose and ids is not None:
            rvecs, tvecs = detector.estimate_pose(corners)
            result = detector.draw_axes(result, corners, rvecs, tvecs)

        # Display info
        marker_count = len(ids) if ids is not None else 0
        cv2.putText(result, f"Dict: {dict_names[current_dict]} | Markers: {marker_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if show_pose:
            cv2.putText(result, "Pose: ON", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ArUco Detection", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif ord('1') <= key <= ord('4'):
            current_dict = key - ord('1')
            detector = ArucoDetector(dict_names[current_dict])
            detector.set_camera_params(frame.shape)
            print(f"Switched to {dict_names[current_dict]}")
        elif key == ord('p'):
            show_pose = not show_pose
        elif key == ord('g'):
            marker = generate_marker(dict_names[current_dict], 0, 300)
            cv2.imwrite("aruco_marker.png", marker)
            print("Saved: aruco_marker.png")

            # Also save board
            board = generate_marker_board(dict_names[current_dict])
            cv2.imwrite("aruco_board.png", board)
            print("Saved: aruco_board.png")
        elif key == ord('s'):
            cv2.imwrite("aruco_detection.jpg", result)
            print("Saved: aruco_detection.jpg")

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with static image.
    """
    print("\n=== ArUco Detection Demo ===\n")

    # Load or generate image
    img = load_aruco_image()

    detector = ArucoDetector('4x4_50')
    detector.set_camera_params(img.shape)

    # Detect markers
    corners, ids, rejected = detector.detect_markers(img)

    # Draw results
    result = detector.draw_markers(img, corners, ids)

    # Pose estimation
    rvecs, tvecs = detector.estimate_pose(corners)
    result = detector.draw_axes(result, corners, rvecs, tvecs)

    marker_count = len(ids) if ids is not None else 0
    print(f"Detected {marker_count} markers")

    if ids is not None:
        print("Marker IDs:", ids.flatten().tolist())

    # Generate sample marker
    marker = generate_marker('4x4_50', 0, 200)
    marker_display = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    cv2.putText(marker_display, "ID: 0", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("ArUco Detection", result)
    cv2.imshow("Sample Marker (4x4_50, ID:0)", marker_display)

    print("\nArUco Applications:")
    print("- Augmented reality")
    print("- Robot navigation")
    print("- Camera calibration")
    print("- Object tracking")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 11: ArUco Marker Detection")
    print("=" * 60)

    try:
        interactive_detector()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
