"""
Application 03: Real-time Image Filters
========================================
Instagram/TikTok style filters using OpenCV.

Techniques Used:
- Custom convolution kernels
- Color manipulation
- Look-Up Tables (LUTs)
- Blending and masking

Official Docs:
- https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class ImageFilters:
    """
    Collection of Instagram-style image filters.
    """

    @staticmethod
    def grayscale(image):
        """Classic black and white."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def sepia(image):
        """Warm vintage sepia tone."""
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        return cv2.transform(image, kernel)

    @staticmethod
    def negative(image):
        """Invert colors."""
        return cv2.bitwise_not(image)

    @staticmethod
    def warm(image):
        """Warm color filter (increase red/yellow)."""
        result = image.copy().astype(np.float32)
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.2, 0, 255)  # Red
        result[:, :, 1] = np.clip(result[:, :, 1] * 1.1, 0, 255)  # Green
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.9, 0, 255)  # Blue
        return result.astype(np.uint8)

    @staticmethod
    def cool(image):
        """Cool color filter (increase blue)."""
        result = image.copy().astype(np.float32)
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.2, 0, 255)  # Blue
        result[:, :, 2] = np.clip(result[:, :, 2] * 0.9, 0, 255)  # Red
        return result.astype(np.uint8)

    @staticmethod
    def vintage(image):
        """Vintage photo effect."""
        # Add sepia
        sepia = ImageFilters.sepia(image)

        # Reduce contrast
        result = cv2.addWeighted(sepia, 0.8, np.full_like(sepia, 40), 0.2, 0)

        # Add vignette
        result = ImageFilters.vignette(result, strength=0.3)

        return result

    @staticmethod
    def vignette(image, strength=0.5):
        """Add vignette effect (dark corners)."""
        rows, cols = image.shape[:2]

        # Create gradient
        x = cv2.getGaussianKernel(cols, cols * 0.5)
        y = cv2.getGaussianKernel(rows, rows * 0.5)
        mask = y @ x.T

        # Normalize
        mask = mask / mask.max()

        # Adjust strength
        mask = mask ** (1 - strength)

        # Apply to each channel
        result = image.copy().astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask

        return result.astype(np.uint8)

    @staticmethod
    def pencil_sketch(image):
        """Pencil sketch effect."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def cartoon(image):
        """Cartoon/comic effect."""
        # Reduce colors with bilateral filter
        color = image.copy()
        for _ in range(2):
            color = cv2.bilateralFilter(color, 9, 300, 300)

        # Get edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 2
        )
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Combine
        return cv2.bitwise_and(color, edges)

    @staticmethod
    def emboss(image):
        """Emboss/3D effect."""
        kernel = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        return cv2.filter2D(image, -1, kernel) + 128

    @staticmethod
    def sharpen(image):
        """Sharpen image."""
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def blur(image, strength=15):
        """Blur effect."""
        return cv2.GaussianBlur(image, (strength, strength), 0)

    @staticmethod
    def edge_glow(image):
        """Glowing edges effect."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None)

        # Create colored edges
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[:, :, 0] = edges  # Blue
        edges_colored[:, :, 1] = 0
        edges_colored[:, :, 2] = edges  # Red (makes purple)

        # Blend with original
        return cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)

    @staticmethod
    def hdr(image):
        """HDR-like effect."""
        # Detail enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Increase saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        hsv = hsv.astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def posterize(image, levels=4):
        """Posterize (reduce color levels)."""
        # Quantize colors
        div = 256 // levels
        return (image // div) * div + div // 2

    @staticmethod
    def pixelate(image, pixel_size=10):
        """Pixelate effect."""
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // pixel_size, h // pixel_size),
                          interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def summer(image):
        """Summer vibes filter."""
        # Increase warmth and contrast
        result = image.copy().astype(np.float32)

        # Warm up
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * 1.05, 0, 255)

        result = result.astype(np.uint8)

        # Increase contrast
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def winter(image):
        """Winter/cold filter."""
        result = image.copy().astype(np.float32)

        # Cool down
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.15, 0, 255)  # Blue
        result[:, :, 2] = np.clip(result[:, :, 2] * 0.9, 0, 255)   # Red

        result = result.astype(np.uint8)

        # Slight desaturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.8, 0, 255)
        hsv = hsv.astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Available filters
FILTERS = {
    '0': ('Original', lambda x: x),
    '1': ('Grayscale', ImageFilters.grayscale),
    '2': ('Sepia', ImageFilters.sepia),
    '3': ('Negative', ImageFilters.negative),
    '4': ('Warm', ImageFilters.warm),
    '5': ('Cool', ImageFilters.cool),
    '6': ('Vintage', ImageFilters.vintage),
    '7': ('Pencil Sketch', ImageFilters.pencil_sketch),
    '8': ('Cartoon', ImageFilters.cartoon),
    '9': ('Emboss', ImageFilters.emboss),
    'a': ('Sharpen', ImageFilters.sharpen),
    'b': ('Blur', ImageFilters.blur),
    'c': ('Edge Glow', ImageFilters.edge_glow),
    'd': ('HDR', ImageFilters.hdr),
    'e': ('Posterize', ImageFilters.posterize),
    'f': ('Pixelate', ImageFilters.pixelate),
    'g': ('Summer', ImageFilters.summer),
    'h': ('Winter', ImageFilters.winter),
    'v': ('Vignette', ImageFilters.vignette),
}


def interactive_filters():
    """
    Interactive filter preview with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== Real-time Image Filters ===")
    print("Press key to apply filter:")
    for key, (name, _) in FILTERS.items():
        print(f"  '{key}' - {name}")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("================================\n")

    current_filter = '0'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply current filter
        name, filter_func = FILTERS[current_filter]
        result = filter_func(frame)

        # Add filter name
        cv2.putText(result, name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Filters", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"filter_{name.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(filename, result)
            print(f"Saved: {filename}")
        elif chr(key) in FILTERS:
            current_filter = chr(key)

    cap.release()
    cv2.destroyAllWindows()


def load_demo_image():
    """
    Load a real image for filter demo, or create one if not available.
    """
    # Try to load sample images
    for sample in ["lena.jpg", "baboon.jpg", "fruits.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            # Resize for consistent display
            img = cv2.resize(img, (400, 300))
            return img

    # Fallback: create colorful test image
    print("No sample image found. Using synthetic demo.")
    print("Run: python curriculum/sample_data/download_samples.py")

    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Gradient background
    for i in range(400):
        img[:, i] = (i * 255 // 400, 150, 255 - i * 255 // 400)

    # Add shapes
    cv2.circle(img, (200, 150), 80, (0, 255, 255), -1)
    cv2.rectangle(img, (50, 200), (150, 280), (255, 100, 0), -1)

    return img


def demo_mode():
    """
    Demo showing all filters on a sample image.
    """
    print("\n=== Filter Demo ===\n")

    # Load real image or create test
    img = load_demo_image()

    # Show all filters
    filter_keys = list(FILTERS.keys())[:9]  # First 9 filters

    rows = []
    for i in range(0, len(filter_keys), 3):
        row_images = []
        for j in range(3):
            if i + j < len(filter_keys):
                key = filter_keys[i + j]
                name, func = FILTERS[key]
                filtered = func(img)
                # Add label
                cv2.putText(filtered, name, (5, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                row_images.append(filtered)

        if row_images:
            rows.append(np.hstack(row_images))

    display = np.vstack(rows)
    cv2.imshow("Filter Gallery", display)

    print("Showing filter gallery...")
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 03: Real-time Image Filters")
    print("=" * 60)

    try:
        interactive_filters()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
