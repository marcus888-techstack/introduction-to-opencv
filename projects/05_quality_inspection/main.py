"""
Project 5: AI-Powered Quality Inspection System
================================================
Industrial defect detection system - detect cracks, scratches, or anomalies
in products.

Key Concepts:
- Image preprocessing (normalization, enhancement)
- Template matching for comparison
- Anomaly detection techniques
- Deep learning for defect classification
- Pass/fail decision logic

Official OpenCV References:
- Image Processing: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
- Template Matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Try to import deep learning libraries
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class DefectDetector:
    """
    Multi-method defect detection system.
    Combines traditional CV techniques with optional deep learning.
    """

    def __init__(self, reference_image=None):
        """
        Initialize detector.

        Args:
            reference_image: Good reference image for comparison
        """
        self.reference = None
        if reference_image is not None:
            self.set_reference(reference_image)

        # Detection thresholds
        self.blob_params = cv2.SimpleBlobDetector_Params()
        self.blob_params.minThreshold = 10
        self.blob_params.maxThreshold = 200
        self.blob_params.filterByArea = True
        self.blob_params.minArea = 50
        self.blob_params.maxArea = 5000
        self.blob_params.filterByCircularity = False
        self.blob_params.filterByConvexity = False
        self.blob_params.filterByInertia = False

        self.blob_detector = cv2.SimpleBlobDetector_create(self.blob_params)

    def set_reference(self, image):
        """Set reference (good) image for comparison."""
        if isinstance(image, str):
            image = cv2.imread(image)
        self.reference = image
        self.reference_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Reference image set")

    def preprocess(self, image):
        """
        Preprocess image for defect detection.

        Steps:
        1. Convert to grayscale
        2. Apply CLAHE for contrast enhancement
        3. Gaussian blur for noise reduction
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE - Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        return blurred

    def detect_by_threshold(self, image, threshold=127):
        """
        Detect anomalies using adaptive thresholding.
        Good for detecting dark spots, scratches.
        """
        preprocessed = self.preprocess(image)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            preprocessed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # Find defect contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        defects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Min defect size
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'type': 'spot',
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })

        return defects

    def detect_by_edges(self, image):
        """
        Detect defects using edge detection.
        Good for detecting cracks and scratches.
        """
        preprocessed = self.preprocess(image)

        # Canny edge detection
        edges = cv2.Canny(preprocessed, 50, 150)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        defects = []
        for contour in contours:
            length = cv2.arcLength(contour, False)
            if length > 30:  # Min crack length
                x, y, w, h = cv2.boundingRect(contour)

                # Classify as crack if elongated
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                if aspect_ratio > 3:
                    defects.append({
                        'type': 'crack',
                        'bbox': (x, y, w, h),
                        'length': length,
                        'contour': contour
                    })

        return defects

    def detect_by_blob(self, image):
        """
        Detect defects using blob detection.
        Good for circular defects like holes, bubbles.
        """
        preprocessed = self.preprocess(image)

        # Invert for dark blob detection
        inverted = cv2.bitwise_not(preprocessed)

        keypoints = self.blob_detector.detect(inverted)

        defects = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            defects.append({
                'type': 'blob',
                'center': (x, y),
                'size': size,
                'bbox': (x - size//2, y - size//2, size, size)
            })

        return defects

    def detect_by_comparison(self, image, threshold=30):
        """
        Detect defects by comparing to reference image.
        Finds differences between test and reference.
        """
        if self.reference is None:
            print("No reference image set")
            return []

        # Resize to match reference
        test = cv2.resize(image, (self.reference.shape[1], self.reference.shape[0]))
        test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(self.reference_gray, test_gray)

        # Threshold difference
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Find defect regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        defects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'type': 'difference',
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })

        return defects

    def detect_all(self, image):
        """
        Run all detection methods and combine results.

        Returns:
            Combined list of all defects found
        """
        all_defects = []

        # Run each method
        all_defects.extend(self.detect_by_threshold(image))
        all_defects.extend(self.detect_by_edges(image))
        all_defects.extend(self.detect_by_blob(image))

        if self.reference is not None:
            all_defects.extend(self.detect_by_comparison(image))

        # Remove duplicates (overlapping detections)
        all_defects = self._remove_duplicates(all_defects)

        return all_defects

    def _remove_duplicates(self, defects, overlap_threshold=0.5):
        """Remove overlapping defect detections."""
        if len(defects) <= 1:
            return defects

        filtered = []
        for defect in defects:
            is_duplicate = False
            bbox = defect.get('bbox')

            if bbox:
                for existing in filtered:
                    existing_bbox = existing.get('bbox')
                    if existing_bbox:
                        iou = self._compute_iou(bbox, existing_bbox)
                        if iou > overlap_threshold:
                            is_duplicate = True
                            break

            if not is_duplicate:
                filtered.append(defect)

        return filtered

    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi

        if wi <= 0 or hi <= 0:
            return 0.0

        intersection = wi * hi
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0


class QualityInspector:
    """
    Main quality inspection system with pass/fail decisions.
    """

    def __init__(self, reference_image=None):
        self.detector = DefectDetector(reference_image)

        # Quality thresholds
        self.max_defects = 3
        self.max_defect_area = 1000
        self.critical_defect_types = ['crack']

    def set_thresholds(self, max_defects=3, max_area=1000, critical_types=None):
        """Set quality thresholds."""
        self.max_defects = max_defects
        self.max_defect_area = max_area
        self.critical_defect_types = critical_types or ['crack']

    def inspect(self, image):
        """
        Perform quality inspection.

        Returns:
            Dict with pass/fail result and details
        """
        # Detect all defects
        defects = self.detector.detect_all(image)

        # Analyze defects
        total_defects = len(defects)
        total_area = sum(d.get('area', d.get('size', 0)**2) for d in defects)
        critical_found = any(d['type'] in self.critical_defect_types for d in defects)

        # Make decision
        passed = True
        fail_reasons = []

        if total_defects > self.max_defects:
            passed = False
            fail_reasons.append(f"Too many defects: {total_defects} > {self.max_defects}")

        if total_area > self.max_defect_area:
            passed = False
            fail_reasons.append(f"Total defect area too large: {total_area} > {self.max_defect_area}")

        if critical_found:
            passed = False
            fail_reasons.append("Critical defect type found")

        return {
            'passed': passed,
            'defects': defects,
            'total_defects': total_defects,
            'total_area': total_area,
            'fail_reasons': fail_reasons,
            'timestamp': datetime.now().isoformat()
        }

    def visualize(self, image, result):
        """Create visualization of inspection result."""
        display = image.copy()

        # Draw defects
        for defect in result['defects']:
            bbox = defect.get('bbox')
            if bbox:
                x, y, w, h = bbox
                color = (0, 0, 255)  # Red for defect
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, defect['type'], (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw result
        status = "PASS" if result['passed'] else "FAIL"
        color = (0, 255, 0) if result['passed'] else (0, 0, 255)

        cv2.putText(display, status, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(display, f"Defects: {result['total_defects']}",
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return display


def create_demo_samples():
    """Create demo images for testing."""
    # Good sample (no defects)
    good = np.ones((400, 400, 3), dtype=np.uint8) * 200
    cv2.rectangle(good, (50, 50), (350, 350), (180, 180, 180), -1)

    # Defective sample (with spots and crack)
    defective = good.copy()
    # Add dark spots
    cv2.circle(defective, (150, 150), 15, (50, 50, 50), -1)
    cv2.circle(defective, (280, 200), 10, (60, 60, 60), -1)
    # Add crack
    pts = np.array([[100, 300], [150, 280], [200, 310], [250, 290], [300, 320]])
    cv2.polylines(defective, [pts], False, (30, 30, 30), 2)

    return good, defective


def run_demo():
    """Run demonstration of quality inspection."""
    print("Quality Inspection Demo")
    print("=" * 40)

    # Create samples
    good_sample, defective_sample = create_demo_samples()

    # Create inspector with good sample as reference
    inspector = QualityInspector()
    inspector.detector.set_reference(good_sample)

    # Inspect good sample
    print("\nInspecting GOOD sample...")
    result_good = inspector.inspect(good_sample)
    print(f"Result: {'PASS' if result_good['passed'] else 'FAIL'}")
    print(f"Defects found: {result_good['total_defects']}")

    # Inspect defective sample
    print("\nInspecting DEFECTIVE sample...")
    result_bad = inspector.inspect(defective_sample)
    print(f"Result: {'PASS' if result_bad['passed'] else 'FAIL'}")
    print(f"Defects found: {result_bad['total_defects']}")
    if result_bad['fail_reasons']:
        print("Reasons:")
        for reason in result_bad['fail_reasons']:
            print(f"  - {reason}")

    # Visualize
    vis_good = inspector.visualize(good_sample, result_good)
    vis_bad = inspector.visualize(defective_sample, result_bad)

    cv2.imshow("Good Sample", vis_good)
    cv2.imshow("Defective Sample", vis_bad)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera_inspection():
    """Run real-time inspection from camera."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    inspector = QualityInspector()

    print("Quality Inspection - Camera Mode")
    print("Controls:")
    print("  R - Set current frame as reference")
    print("  SPACE - Inspect current frame")
    print("  Q - Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, "Press R to set reference, SPACE to inspect",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Quality Inspection", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            inspector.detector.set_reference(frame)
            print("Reference set!")

        elif key == ord(' '):
            result = inspector.inspect(frame)
            vis = inspector.visualize(frame, result)
            cv2.imshow("Inspection Result", vis)

            print(f"\nResult: {'PASS' if result['passed'] else 'FAIL'}")
            print(f"Defects: {result['total_defects']}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quality Inspection System")
    parser.add_argument("--image", type=str, help="Inspect image file")
    parser.add_argument("--reference", type=str, help="Reference image")
    parser.add_argument("--camera", action="store_true", help="Use camera")
    parser.add_argument("--demo", action="store_true", help="Run demo")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.camera:
        run_camera_inspection()
    elif args.image:
        inspector = QualityInspector()
        if args.reference:
            inspector.detector.set_reference(args.reference)

        image = cv2.imread(args.image)
        result = inspector.inspect(image)
        vis = inspector.visualize(image, result)

        print(f"Result: {'PASS' if result['passed'] else 'FAIL'}")
        print(f"Defects: {result['total_defects']}")

        cv2.imshow("Result", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Quality Inspection System")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python main.py --demo                    # Run demo")
        print("  python main.py --camera                  # Use webcam")
        print("  python main.py --image FILE              # Inspect image")
        print("  python main.py --image FILE --reference REF  # With reference")
