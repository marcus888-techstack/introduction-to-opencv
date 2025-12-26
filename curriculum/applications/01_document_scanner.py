"""
Application 01: Document Scanner
================================
Scan documents like CamScanner/Adobe Scan using OpenCV.

Techniques Used:
- Edge detection (Canny)
- Contour detection
- Perspective transform
- Adaptive thresholding

Official Docs:
- https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image, get_video, SAMPLE_DIR


def order_points(pts):
    """
    Order points in: top-left, top-right, bottom-right, bottom-left order.
    This is required for perspective transform.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum: top-left has smallest sum, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # Difference: top-right has smallest diff, bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def four_point_transform(image, pts):
    """
    Apply perspective transform to get bird's eye view of document.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Compute height of new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination points for bird's eye view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Compute perspective transform matrix and apply
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def find_document_contour(image):
    """
    Find the largest 4-point contour (document edges).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None

    for contour in contours[:5]:  # Check top 5 largest
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If 4 points, we found our document
        if len(approx) == 4:
            document_contour = approx
            break

    return document_contour, edges


def enhance_document(image):
    """
    Enhance scanned document for better readability.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for clean black & white
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Optional: Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    return sharpened


def scan_document(image):
    """
    Main document scanning pipeline.
    """
    # Keep original for final transform
    original = image.copy()

    # Resize for faster processing (keep aspect ratio)
    height, width = image.shape[:2]
    ratio = height / 500.0
    resized = cv2.resize(image, (int(width / ratio), 500))

    # Find document contour
    contour, edges = find_document_contour(resized)

    if contour is None:
        print("No document found! Using full image.")
        return image, None, edges

    # Scale contour back to original size
    contour = contour.reshape(4, 2) * ratio

    # Apply perspective transform
    scanned = four_point_transform(original, contour)

    # Enhance the scanned document
    enhanced = enhance_document(scanned)

    return enhanced, contour, edges


def load_demo_document():
    """
    Load a real document image for demo, or create one if not available.
    """
    # Try to load sudoku image (good document-like image)
    img = get_image("sudoku.png")
    if img is not None:
        print("Using sample image: sudoku.png")
        # Add perspective distortion to simulate a photo of document
        h, w = img.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[50, 30], [w-30, 50], [20, h-40], [w-50, h-20]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # Create background
        result = np.ones((h + 100, w + 100, 3), dtype=np.uint8) * 180
        noise = np.random.randint(0, 30, result.shape, dtype=np.uint8)
        result = cv2.add(result, noise)

        # Warp document
        warped = cv2.warpPerspective(img, M, (w + 100, h + 100))
        mask = cv2.warpPerspective(np.ones_like(img) * 255, M, (w + 100, h + 100))

        # Blend
        result = np.where(mask > 0, warped, result)
        return result

    # Try newspaper image
    img = get_image("newspaper.jpg")
    if img is not None:
        print("Using sample image: newspaper.jpg")
        return img

    # Try text image
    img = get_image("imageTextN.png")
    if img is not None:
        print("Using sample image: imageTextN.png")
        # Convert and add background
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        h, w = img.shape[:2]
        result = np.ones((h + 100, w + 100, 3), dtype=np.uint8) * 200
        result[50:50+h, 50:50+w] = img
        return result

    # Fallback: create synthetic document
    print("No document sample found. Creating synthetic document.")
    print("Run: python curriculum/sample_data/download_samples.py")

    img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    doc_pts = np.array([[150, 80], [650, 120], [600, 520], [100, 480]], dtype=np.int32)
    cv2.fillPoly(img, [doc_pts], (255, 255, 255))

    cv2.putText(img, "INVOICE", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(img, "Item: OpenCV Tutorial", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)
    cv2.putText(img, "Price: $49.99", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)

    return img


def interactive_scanner():
    """
    Interactive document scanner with webcam.
    Press 's' to scan, 'q' to quit.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo image.")
        demo_mode()
        return

    print("\n=== Interactive Document Scanner ===")
    print("Controls:")
    print("  's' - Scan document")
    print("  'q' - Quit")
    print("====================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Find document contour for preview
        height, width = frame.shape[:2]
        ratio = height / 500.0
        resized = cv2.resize(frame, (int(width / ratio), 500))
        contour, _ = find_document_contour(resized)

        # Draw detected contour
        display = frame.copy()
        if contour is not None:
            contour_scaled = (contour.reshape(4, 2) * ratio).astype(np.int32)
            cv2.drawContours(display, [contour_scaled], -1, (0, 255, 0), 3)
            cv2.putText(display, "Document detected! Press 's' to scan",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Position document in frame",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Document Scanner", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and contour is not None:
            # Scan document
            scanned, _, _ = scan_document(frame)
            cv2.imshow("Scanned Document", scanned)
            print("Document scanned! Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow("Scanned Document")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo mode with real or synthetic document.
    """
    print("\n=== Document Scanner Demo ===\n")

    # Load demo document (real image preferred)
    original = load_demo_document()

    # Scan document
    scanned, contour, edges = scan_document(original)

    # Visualization
    display_original = original.copy()
    if contour is not None:
        cv2.drawContours(display_original, [contour.astype(np.int32)], -1, (0, 255, 0), 3)
        for point in contour.astype(np.int32):
            cv2.circle(display_original, tuple(point), 10, (0, 0, 255), -1)

    # Show results
    cv2.imshow("1. Original with detected corners", display_original)
    cv2.imshow("2. Edge Detection", edges)
    cv2.imshow("3. Scanned Document", scanned)

    print("Steps:")
    print("1. Detect edges using Canny")
    print("2. Find largest 4-point contour")
    print("3. Apply perspective transform")
    print("4. Enhance with adaptive thresholding")
    print("\nPress any key to close...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 01: Document Scanner")
    print("=" * 60)

    # Try webcam first, fall back to demo
    try:
        interactive_scanner()
    except Exception as e:
        print(f"Camera error: {e}")
        demo_mode()
