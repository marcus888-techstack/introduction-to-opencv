"""
Application 05: Object Counter
==============================
Count objects in images using contours and connected components.

Techniques Used:
- Thresholding (Otsu, adaptive)
- Morphological operations
- Contour detection
- Connected components
- Watershed for touching objects

Official Docs:
- https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class ObjectCounter:
    """
    Count objects in images using various methods.
    """

    def __init__(self, min_area=100, max_area=None):
        self.min_area = min_area
        self.max_area = max_area

    def preprocess(self, image, method='otsu'):
        """
        Preprocess image for counting.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding
        if method == 'otsu':
            _, thresh = cv2.threshold(blurred, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            thresh = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
        else:  # simple
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        # Clean up with morphology
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        return thresh

    def count_contours(self, image, method='otsu'):
        """
        Count objects using contour detection.
        """
        thresh = self.preprocess(image, method)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                if self.max_area is None or area <= self.max_area:
                    valid_contours.append(cnt)

        return valid_contours, thresh

    def count_connected_components(self, image, method='otsu'):
        """
        Count objects using connected components.
        """
        thresh = self.preprocess(image, method)

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        # Filter by area (skip background label 0)
        valid_labels = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                if self.max_area is None or area <= self.max_area:
                    valid_labels.append(i)

        return len(valid_labels), labels, stats, centroids

    def count_watershed(self, image):
        """
        Count touching objects using watershed algorithm.
        Good for overlapping/touching objects.
        """
        # Preprocess
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background (dilate)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Distance transform for sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(),
                                   255, 0)
        sure_fg = np.uint8(sure_fg)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add 1 to all labels so background is 1 instead of 0
        markers = markers + 1

        # Mark unknown region as 0
        markers[unknown == 255] = 0

        # Apply watershed
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()

        markers = cv2.watershed(image_color, markers)

        # Count unique markers (excluding background and boundary)
        unique_markers = np.unique(markers)
        count = len([m for m in unique_markers if m > 1])

        return count, markers

    def draw_results(self, image, contours=None, labels=None, stats=None,
                     centroids=None, markers=None):
        """
        Draw counting results on image.
        """
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        if contours is not None:
            for i, cnt in enumerate(contours):
                # Draw contour
                cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

                # Get centroid
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(result, str(i + 1), (cx - 10, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        elif labels is not None and stats is not None and centroids is not None:
            for i in range(1, labels.max() + 1):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_area:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    cx, cy = centroids[i]

                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(result, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        elif markers is not None:
            # Color each region differently
            colors = np.random.randint(0, 255, (markers.max() + 1, 3))
            colors[0] = [0, 0, 0]  # Background
            colors[1] = [128, 128, 128]  # Boundary

            for i in range(2, markers.max() + 1):
                mask = (markers == i).astype(np.uint8)
                result[mask == 1] = colors[i]

        return result


def load_countable_image():
    """
    Load a real image with countable objects, or create one if not available.
    """
    # Try to load images with countable objects
    for sample in ["coins.jpg", "pills.jpg", "nuts.jpg", "fruits.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Fallback to synthetic
    print("No countable sample found. Using synthetic circles.")
    print("Run: python curriculum/sample_data/download_samples.py")
    return None


def create_test_image(num_objects=10, touching=False):
    """
    Create a test image with random objects (fallback).
    """
    img = np.ones((400, 600), dtype=np.uint8) * 255

    np.random.seed(42)

    if touching:
        # Create touching circles (coins)
        positions = [(100, 100), (140, 100), (180, 100),
                    (120, 140), (160, 140),
                    (300, 200), (340, 200), (380, 200),
                    (320, 240), (360, 240)]
        for x, y in positions[:num_objects]:
            cv2.circle(img, (x, y), 30, 0, -1)
    else:
        # Create random non-touching circles
        for _ in range(num_objects):
            x = np.random.randint(50, 550)
            y = np.random.randint(50, 350)
            r = np.random.randint(15, 40)
            cv2.circle(img, (x, y), r, 0, -1)

    return img


def interactive_counter():
    """
    Interactive object counter with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== Object Counter ===")
    print("Controls:")
    print("  '1' - Contour method")
    print("  '2' - Connected components")
    print("  '3' - Watershed (for touching objects)")
    print("  'o' - Otsu thresholding")
    print("  'a' - Adaptive thresholding")
    print("  '+' - Increase min area")
    print("  '-' - Decrease min area")
    print("  's' - Save screenshot")
    print("  'q' - Quit")
    print("======================\n")

    counter = ObjectCounter(min_area=500)
    method = 'contours'
    thresh_method = 'otsu'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if method == 'contours':
            contours, thresh = counter.count_contours(frame, thresh_method)
            result = counter.draw_results(frame, contours=contours)
            count = len(contours)
        elif method == 'connected':
            count, labels, stats, centroids = counter.count_connected_components(
                frame, thresh_method
            )
            result = counter.draw_results(frame, labels=labels, stats=stats,
                                         centroids=centroids)
            thresh = counter.preprocess(frame, thresh_method)
        else:  # watershed
            count, markers = counter.count_watershed(frame)
            result = counter.draw_results(frame, markers=markers)
            thresh = counter.preprocess(frame, 'otsu')

        # Display info
        info = f"Method: {method} | Count: {count} | Min Area: {counter.min_area}"
        cv2.putText(result, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Object Counter", result)
        cv2.imshow("Threshold", thresh)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            method = 'contours'
        elif key == ord('2'):
            method = 'connected'
        elif key == ord('3'):
            method = 'watershed'
        elif key == ord('o'):
            thresh_method = 'otsu'
        elif key == ord('a'):
            thresh_method = 'adaptive'
        elif key == ord('+') or key == ord('='):
            counter.min_area += 100
        elif key == ord('-'):
            counter.min_area = max(50, counter.min_area - 100)
        elif key == ord('s'):
            cv2.imwrite("count_result.jpg", result)
            print("Saved: count_result.jpg")

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with real images or test images.
    """
    print("\n=== Object Counter Demo ===\n")

    # Try to load real countable image
    real_img = load_countable_image()

    counter = ObjectCounter(min_area=100)

    if real_img is not None:
        # Use real image
        print("\nCounting objects in real image...")
        contours, thresh = counter.count_contours(real_img)
        result = counter.draw_results(real_img, contours=contours)
        print(f"Found {len(contours)} objects using contours")

        cv2.putText(result, f"Objects: {len(contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Object Counting Demo", result)
    else:
        # Fallback: create test images
        img_separate = create_test_image(8, touching=False)
        img_touching = create_test_image(10, touching=True)

        # Count separate objects
        contours, thresh = counter.count_contours(img_separate)
        result1 = counter.draw_results(img_separate, contours=contours)
        print(f"Separate objects: Found {len(contours)} objects using contours")

        # Count touching objects with watershed
        count, markers = counter.count_watershed(img_touching)
        result2 = counter.draw_results(img_touching, markers=markers)
        print(f"Touching objects: Found {count} objects using watershed")

        # Display
        cv2.putText(result1, f"Contours: {len(contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result2, f"Watershed: {count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        display = np.hstack([
            cv2.cvtColor(result1, cv2.COLOR_GRAY2BGR) if len(result1.shape) == 2 else result1,
            result2
        ])

        cv2.imshow("Object Counting Demo", display)

    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def count_coins(image_path):
    """
    Example: Count coins in an image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    counter = ObjectCounter(min_area=1000)

    # Try watershed for potentially touching coins
    count, markers = counter.count_watershed(img)

    result = counter.draw_results(img, markers=markers)
    cv2.putText(result, f"Coins: {count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Coin Counter", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return count


if __name__ == "__main__":
    print("=" * 60)
    print("Application 05: Object Counter")
    print("=" * 60)

    try:
        interactive_counter()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
