"""
Application 07: QR Code and Barcode Reader
==========================================
Detect and decode QR codes and barcodes in images/video.

Techniques Used:
- QRCodeDetector
- Barcode detection
- Real-time decoding

Official Docs:
- https://docs.opencv.org/4.x/de/dc3/classcv_1_1QRCodeDetector.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class QRBarcodeReader:
    """
    QR code and barcode detection and decoding.
    """

    def __init__(self):
        # QR code detector
        self.qr_detector = cv2.QRCodeDetector()

        # Try to use barcode detector if available (OpenCV 4.5.3+)
        try:
            self.barcode_detector = cv2.barcode.BarcodeDetector()
            self.has_barcode = True
        except AttributeError:
            self.has_barcode = False
            print("Note: Barcode detector not available (requires OpenCV 4.5.3+)")

    def detect_qr(self, image):
        """
        Detect and decode QR codes in image.
        Returns: (decoded_data, points, straight_qr)
        """
        # Detect and decode
        data, points, straight_qr = self.qr_detector.detectAndDecode(image)

        return data, points, straight_qr

    def detect_multi_qr(self, image):
        """
        Detect multiple QR codes in image.
        Returns: list of (data, points) tuples
        """
        results = []

        # Try multi-detection
        retval, decoded_info, points, straight_qr = self.qr_detector.detectAndDecodeMulti(image)

        if retval and decoded_info is not None:
            for i, data in enumerate(decoded_info):
                if data:
                    pts = points[i] if points is not None else None
                    results.append((data, pts))

        return results

    def detect_barcode(self, image):
        """
        Detect and decode barcodes in image.
        Returns: list of (data, type, points) tuples
        """
        if not self.has_barcode:
            return []

        results = []

        # Detect barcodes
        retval, decoded_info, decoded_type, points = self.barcode_detector.detectAndDecode(image)

        if retval and decoded_info is not None:
            for i, data in enumerate(decoded_info):
                if data:
                    bc_type = decoded_type[i] if decoded_type is not None else "Unknown"
                    pts = points[i] if points is not None else None
                    results.append((data, bc_type, pts))

        return results

    def draw_detection(self, image, points, data, color=(0, 255, 0)):
        """
        Draw detection result on image.
        """
        result = image.copy()

        if points is not None:
            points = points.astype(int)

            # Draw polygon around QR/barcode
            n = len(points)
            for i in range(n):
                pt1 = tuple(points[i])
                pt2 = tuple(points[(i + 1) % n])
                cv2.line(result, pt1, pt2, color, 3)

            # Draw data text
            if data:
                x, y = points[0]
                cv2.putText(result, data[:50], (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result

    def process_image(self, image):
        """
        Process image and detect all QR codes and barcodes.
        """
        result = image.copy()
        detections = []

        # Detect QR codes
        qr_results = self.detect_multi_qr(image)
        for data, points in qr_results:
            detections.append(('QR', data, points))
            if points is not None:
                result = self.draw_detection(result, points, f"QR: {data}", (0, 255, 0))

        # Detect barcodes
        bc_results = self.detect_barcode(image)
        for data, bc_type, points in bc_results:
            detections.append(('Barcode', data, points))
            if points is not None:
                result = self.draw_detection(result, points, f"{bc_type}: {data}", (255, 0, 0))

        return result, detections


def generate_qr_code(data, size=200):
    """
    Generate a simple QR code image (for demo purposes).
    Note: This creates a placeholder - real QR generation needs qrcode library.
    """
    # Create placeholder QR-like image
    img = np.ones((size, size), dtype=np.uint8) * 255

    # Simple pattern (not a real QR code)
    block = size // 21
    for i in range(21):
        for j in range(21):
            if (i < 7 and j < 7) or (i < 7 and j > 13) or (i > 13 and j < 7):
                # Position markers
                if (i in [0, 6] or j in [0, 6]) or (2 <= i <= 4 and 2 <= j <= 4):
                    img[i*block:(i+1)*block, j*block:(j+1)*block] = 0
            elif np.random.random() > 0.5:
                img[i*block:(i+1)*block, j*block:(j+1)*block] = 0

    return img


def load_qr_image():
    """
    Load a real QR code image or create one for demo.
    """
    # Try to load QR sample
    for sample in ["qr_code.png", "qrcode.png"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Create demo QR code
    print("No QR sample found. Creating demo QR pattern.")
    print("For real QR codes, use: pip install qrcode[pil]")

    qr = generate_qr_code("https://opencv.org", 300)

    # Add to a larger image with background
    img = np.ones((400, 500, 3), dtype=np.uint8) * 240
    qr_color = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    img[50:350, 100:400] = qr_color

    cv2.putText(img, "Demo QR Pattern", (150, 380),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    return img


def interactive_reader():
    """
    Interactive QR/barcode reader with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== QR Code & Barcode Reader ===")
    print("Controls:")
    print("  's' - Save screenshot")
    print("  'q' - Quit")
    print("================================\n")
    print("Point camera at QR codes or barcodes to scan them.")

    reader = QRBarcodeReader()
    last_detection = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result, detections = reader.process_image(frame)

        # Display detection count
        cv2.putText(result, f"Detections: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Print new detections
        for det_type, data, _ in detections:
            if data != last_detection:
                print(f"[{det_type}] Detected: {data}")
                last_detection = data

        cv2.imshow("QR/Barcode Reader", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = "qr_scan_result.jpg"
            cv2.imwrite(filename, result)
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with sample QR code image.
    """
    print("\n=== QR Code Reader Demo ===\n")

    # Load QR image
    img = load_qr_image()

    reader = QRBarcodeReader()

    # Process image
    result, detections = reader.process_image(img)

    print(f"Found {len(detections)} codes:")
    for det_type, data, _ in detections:
        print(f"  [{det_type}] {data}")

    if len(detections) == 0:
        print("  No codes detected (demo pattern is not a real QR code)")
        print("  Try with a real QR code image or use webcam mode")

    cv2.imshow("QR Code Demo", result)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 07: QR Code & Barcode Reader")
    print("=" * 60)

    try:
        interactive_reader()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
