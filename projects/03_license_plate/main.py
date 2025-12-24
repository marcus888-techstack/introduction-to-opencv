"""
Project 3: License Plate Recognition (ANPR)
============================================
Automatic Number Plate Recognition system used in parking lots, toll booths,
and security systems.

Key Concepts:
- License plate localization (detection)
- Character segmentation
- OCR on plate region
- Real-time video processing
- Database logging

Official OpenCV References:
- Object Detection: https://docs.opencv.org/4.x/d5/d54/group__objdetect.html
- DNN Module: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import re

# Try to import OCR library
try:
    import easyocr
    OCR_AVAILABLE = True
    reader = None  # Lazy initialization
except ImportError:
    OCR_AVAILABLE = False
    print("EasyOCR not installed. Install with: pip install easyocr")

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Ultralytics not installed. Using OpenCV-based detection.")
    print("Install with: pip install ultralytics")


def get_ocr_reader():
    """Lazy initialization of EasyOCR reader."""
    global reader
    if reader is None and OCR_AVAILABLE:
        print("Initializing OCR engine...")
        reader = easyocr.Reader(['en'])
    return reader


class LicensePlateDetector:
    """Detect license plates using various methods."""

    def __init__(self, method='cascade'):
        """
        Initialize detector.

        Args:
            method: 'cascade' for Haar Cascade, 'yolo' for YOLOv8
        """
        self.method = method

        if method == 'cascade':
            # Use Haar Cascade (built-in, no extra download needed)
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
        elif method == 'yolo' and YOLO_AVAILABLE:
            # Use YOLOv8 for better accuracy
            # Note: This uses the general YOLO model - for production,
            # you'd train a custom model on license plates
            self.model = YOLO('yolov8n.pt')

    def detect_plate_cascade(self, image):
        """Detect plates using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect plates
        plates = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 30)
        )

        return [(x, y, w, h) for (x, y, w, h) in plates]

    def detect_plate_contour(self, image):
        """
        Detect plates using contour analysis.
        Works well for standard rectangular plates.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edges = cv2.Canny(filtered, 30, 200)

        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_candidates = []

        for contour in contours:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            # License plates are typically rectangles (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                # Check aspect ratio (plates are wider than tall)
                aspect_ratio = w / float(h)
                if 2.0 < aspect_ratio < 6.0:
                    # Check size
                    if w > 60 and h > 20:
                        plate_candidates.append((x, y, w, h))

        return plate_candidates

    def detect(self, image):
        """
        Detect license plates in image.

        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if self.method == 'cascade':
            plates = self.detect_plate_cascade(image)
            if not plates:
                # Fallback to contour method
                plates = self.detect_plate_contour(image)
            return plates
        else:
            return self.detect_plate_contour(image)


def preprocess_plate(plate_img):
    """
    Preprocess plate image for better OCR results.

    Steps:
    1. Resize to standard width
    2. Convert to grayscale
    3. Apply adaptive thresholding
    4. Denoise
    """
    # Resize to standard width
    height, width = plate_img.shape[:2]
    new_width = 400
    new_height = int(height * (new_width / width))
    resized = cv2.resize(plate_img, (new_width, new_height))

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19, 9
    )

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned, resized


def extract_plate_text(plate_img):
    """
    Extract text from license plate image.

    Args:
        plate_img: Cropped plate image (BGR)

    Returns:
        Tuple of (text, confidence)
    """
    if not OCR_AVAILABLE:
        return "OCR_NOT_AVAILABLE", 0.0

    reader = get_ocr_reader()

    # Preprocess
    processed, original = preprocess_plate(plate_img)

    # Try OCR on both original and processed
    results = reader.readtext(original)

    if not results:
        # Try with processed image
        results = reader.readtext(processed)

    if not results:
        return "", 0.0

    # Combine all detected text
    full_text = ""
    total_conf = 0.0

    for (bbox, text, conf) in results:
        # Clean text - keep only alphanumeric
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        if cleaned:
            full_text += cleaned
            total_conf += conf

    avg_conf = total_conf / len(results) if results else 0.0

    return full_text, avg_conf


def validate_plate_format(text, country='general'):
    """
    Basic validation of license plate format.

    Args:
        text: Extracted plate text
        country: Country format to validate

    Returns:
        Boolean indicating if format is valid
    """
    if not text or len(text) < 4:
        return False

    # General check: should have mix of letters and numbers
    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)

    return has_letters and has_numbers


class PlateLogger:
    """Log detected license plates."""

    def __init__(self, log_dir="plate_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Today's log file
        self.log_file = self.log_dir / f"plates_{datetime.now().strftime('%Y-%m-%d')}.csv"

        # Track plates seen recently (avoid duplicates)
        self.recent_plates = {}
        self.cooldown = 30  # seconds before logging same plate again

    def log_plate(self, plate_text, confidence, image=None):
        """
        Log a detected plate.

        Args:
            plate_text: Detected plate number
            confidence: OCR confidence
            image: Optional plate image to save
        """
        now = datetime.now()

        # Check cooldown
        if plate_text in self.recent_plates:
            last_seen = self.recent_plates[plate_text]
            if (now - last_seen).seconds < self.cooldown:
                return False

        self.recent_plates[plate_text] = now
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # Write to log
        with open(self.log_file, 'a') as f:
            f.write(f"{plate_text},{confidence:.2f},{timestamp}\n")

        # Save image if provided
        if image is not None:
            img_path = self.log_dir / f"{plate_text}_{now.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(img_path), image)

        print(f"Logged: {plate_text} (conf: {confidence:.2f}) at {timestamp}")
        return True


def process_image(image_path, debug=False):
    """
    Process a single image for license plate recognition.

    Args:
        image_path: Path to image file
        debug: Show visualization

    Returns:
        List of detected plates with their text
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return []

    detector = LicensePlateDetector()
    results = []

    # Detect plates
    plates = detector.detect(image)
    print(f"Detected {len(plates)} potential plate(s)")

    display = image.copy()

    for i, (x, y, w, h) in enumerate(plates):
        # Extract plate region
        plate_img = image[y:y+h, x:x+w]

        # Run OCR
        text, confidence = extract_plate_text(plate_img)

        if text and validate_plate_format(text):
            results.append({
                'text': text,
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'image': plate_img
            })

            # Draw on display image
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, f"{text} ({confidence:.0%})",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)

    if debug:
        cv2.imshow("License Plate Detection", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def process_video(video_source=0, log_plates=True):
    """
    Process video stream for license plate recognition.

    Args:
        video_source: Camera ID or video file path
        log_plates: Whether to log detected plates
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Could not open video source: {video_source}")
        return

    detector = LicensePlateDetector()
    logger = PlateLogger() if log_plates else None

    print("License Plate Recognition - Video Mode")
    print("Press Q to quit")

    # Process every N frames for performance
    frame_count = 0
    process_every = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Process frame
        if frame_count % process_every == 0:
            plates = detector.detect(frame)

            for (x, y, w, h) in plates:
                plate_img = frame[y:y+h, x:x+w]
                text, confidence = extract_plate_text(plate_img)

                if text and validate_plate_format(text) and confidence > 0.3:
                    # Draw green box for valid plate
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(display, f"{text} ({confidence:.0%})",
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.9, (0, 255, 0), 2)

                    # Log plate
                    if logger:
                        logger.log_plate(text, confidence, plate_img)
                else:
                    # Draw yellow box for potential plate
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Show status
        cv2.putText(display, "License Plate Recognition",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ANPR System", display)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_demo_plate():
    """Create a simple demo plate image for testing."""
    # Create plate background
    plate = np.ones((100, 300, 3), dtype=np.uint8) * 255

    # Add border
    cv2.rectangle(plate, (5, 5), (295, 95), (0, 0, 0), 2)

    # Add text
    cv2.putText(plate, "ABC 1234", (30, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    return plate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="License Plate Recognition System")
    parser.add_argument("--image", type=str, help="Process image file")
    parser.add_argument("--video", type=str, help="Process video file")
    parser.add_argument("--camera", action="store_true", help="Use webcam")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--debug", action="store_true", help="Show debug output")

    args = parser.parse_args()

    if args.demo:
        print("Creating demo plate image...")
        demo_plate = create_demo_plate()

        # Add plate to a "scene"
        scene = np.ones((400, 600, 3), dtype=np.uint8) * 128
        scene[150:250, 150:450] = demo_plate

        # Save and process
        demo_path = Path("output/demo_plate.jpg")
        demo_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(demo_path), scene)
        print(f"Created: {demo_path}")

        results = process_image(demo_path, debug=True)

        for r in results:
            print(f"Detected: {r['text']} (confidence: {r['confidence']:.2%})")

    elif args.image:
        results = process_image(args.image, debug=args.debug)
        for r in results:
            print(f"Plate: {r['text']} (confidence: {r['confidence']:.2%})")

    elif args.video:
        process_video(args.video)

    elif args.camera:
        process_video(0)

    else:
        print("License Plate Recognition System")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python main.py --demo           # Run with demo image")
        print("  python main.py --camera         # Use webcam")
        print("  python main.py --image FILE     # Process image")
        print("  python main.py --video FILE     # Process video")
