"""
Project 1: Smart Document Scanner with OCR
==========================================
A mobile-scanner-like app that digitizes documents with automatic edge detection
and text extraction.

Key Concepts:
- Edge detection (Canny algorithm)
- Contour detection and document boundary finding
- Perspective transformation (4-point warp)
- Image preprocessing for OCR
- Text extraction with EasyOCR

Official OpenCV References:
- Contours: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
- Perspective Transform: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
- Canny Edge: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
"""

import cv2
import numpy as np
from pathlib import Path

# Optional: OCR support
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("EasyOCR not installed. OCR features disabled.")
    print("Install with: pip install easyocr")


def order_points(pts):
    """
    Order points in consistent order: top-left, top-right, bottom-right, bottom-left.

    This is crucial for perspective transformation to work correctly.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    """
    Apply perspective transformation to obtain a top-down view of the document.

    Args:
        image: Source image
        pts: Four corner points of the document

    Returns:
        Warped image with bird's eye view of the document
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image (max of top and bottom edge)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of new image (max of left and right edge)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def find_document_contour(image, debug=False):
    """
    Find the largest quadrilateral contour (document boundary) in the image.

    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Detect edges using Canny
    4. Find contours
    5. Filter for 4-point contours (rectangles)

    Args:
        image: Source image
        debug: If True, show intermediate steps

    Returns:
        4 corner points of the document, or None if not found
    """
    # Resize for faster processing
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 75, 200)

    if debug:
        cv2.imshow("1. Grayscale", gray)
        cv2.imshow("2. Blurred", blurred)
        cv2.imshow("3. Edges", edged)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the approximated contour has 4 points, we found our document
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is None:
        return None

    # Scale back to original image size
    return doc_contour.reshape(4, 2) * ratio


def enhance_scanned_document(image):
    """
    Enhance the scanned document for better readability.

    Applies:
    - Adaptive thresholding for clean black/white output
    - Optional sharpening
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    # This creates a clean, high-contrast document
    enhanced = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return enhanced


def extract_text(image, languages=['en']):
    """
    Extract text from the document using EasyOCR.

    Args:
        image: Preprocessed document image
        languages: List of languages to detect

    Returns:
        Extracted text as string
    """
    if not OCR_AVAILABLE:
        return "OCR not available. Install easyocr."

    reader = easyocr.Reader(languages)

    # EasyOCR works best with BGR or grayscale images
    if len(image.shape) == 2:
        # Grayscale - convert to BGR for consistency
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    results = reader.readtext(image)

    # Combine all detected text
    text_lines = [result[1] for result in results]
    return '\n'.join(text_lines)


def scan_document(image_path, output_dir="output", extract_ocr=True, debug=False):
    """
    Main function to scan a document from an image.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save results
        extract_ocr: Whether to extract text
        debug: Show intermediate processing steps

    Returns:
        Dictionary with scanned image and extracted text
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Processing: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Find document contour
    doc_contour = find_document_contour(image, debug=debug)

    if doc_contour is None:
        print("Could not detect document boundary. Using full image.")
        scanned = image.copy()
    else:
        print("Document boundary detected!")

        # Draw contour on original for visualization
        if debug:
            contour_img = image.copy()
            cv2.drawContours(contour_img, [doc_contour.astype(int)], -1, (0, 255, 0), 3)
            cv2.imshow("4. Detected Document", contour_img)

        # Apply perspective transform
        scanned = four_point_transform(image, doc_contour)

    # Enhance the document
    enhanced = enhance_scanned_document(scanned)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save results
    input_name = Path(image_path).stem
    scanned_path = output_path / f"{input_name}_scanned.jpg"
    enhanced_path = output_path / f"{input_name}_enhanced.jpg"

    cv2.imwrite(str(scanned_path), scanned)
    cv2.imwrite(str(enhanced_path), enhanced)
    print(f"Saved: {scanned_path}")
    print(f"Saved: {enhanced_path}")

    # Extract text if requested
    extracted_text = ""
    if extract_ocr and OCR_AVAILABLE:
        print("Extracting text...")
        extracted_text = extract_text(scanned)

        # Save text to file
        text_path = output_path / f"{input_name}_text.txt"
        with open(text_path, 'w') as f:
            f.write(extracted_text)
        print(f"Saved: {text_path}")

    if debug:
        cv2.imshow("5. Scanned Document", scanned)
        cv2.imshow("6. Enhanced Document", enhanced)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "scanned": scanned,
        "enhanced": enhanced,
        "text": extracted_text
    }


def scan_from_camera(camera_id=0):
    """
    Interactive document scanning from webcam.

    Controls:
    - SPACE: Capture and process current frame
    - Q: Quit
    """
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Document Scanner - Camera Mode")
    print("Controls:")
    print("  SPACE - Capture and scan document")
    print("  Q     - Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Try to find document in real-time
        display = frame.copy()
        doc_contour = find_document_contour(frame)

        if doc_contour is not None:
            # Draw detected document boundary
            cv2.drawContours(display, [doc_contour.astype(int)], -1, (0, 255, 0), 2)
            cv2.putText(display, "Document Detected - Press SPACE to scan",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Position document in view",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Document Scanner", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Capture and process
            if doc_contour is not None:
                scanned = four_point_transform(frame, doc_contour)
                enhanced = enhance_scanned_document(scanned)

                cv2.imshow("Scanned Document", scanned)
                cv2.imshow("Enhanced Document", enhanced)

                # Save automatically
                timestamp = Path("output") / f"scan_{cv2.getTickCount()}.jpg"
                timestamp.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(timestamp), enhanced)
                print(f"Saved: {timestamp}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Demo with sample image
def create_demo_image():
    """Create a sample document image for testing."""
    # Create a white document on dark background
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark background

    # Draw a tilted white rectangle (document)
    pts = np.array([[150, 100], [650, 80], [680, 500], [120, 520]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # Add some text to the document
    cv2.putText(img, "Sample Document", (200, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "Line 1: Hello World", (180, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Line 2: OpenCV is great!", (180, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Line 3: Document Scanner", (180, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Document Scanner with OCR")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--camera", action="store_true", help="Use camera mode")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample image")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR extraction")

    args = parser.parse_args()

    if args.camera:
        scan_from_camera()
    elif args.demo:
        # Create and save demo image
        demo_img = create_demo_image()
        demo_path = Path("output/demo_input.jpg")
        demo_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(demo_path), demo_img)
        print(f"Created demo image: {demo_path}")

        # Process demo image
        result = scan_document(demo_path, extract_ocr=not args.no_ocr, debug=args.debug)

        if result["text"]:
            print("\n--- Extracted Text ---")
            print(result["text"])
    elif args.image:
        result = scan_document(args.image, extract_ocr=not args.no_ocr, debug=args.debug)

        if result["text"]:
            print("\n--- Extracted Text ---")
            print(result["text"])
    else:
        print("Usage:")
        print("  python main.py --demo           # Run with demo image")
        print("  python main.py --camera         # Use webcam")
        print("  python main.py --image FILE     # Process specific image")
        print("  python main.py --debug          # Show processing steps")
