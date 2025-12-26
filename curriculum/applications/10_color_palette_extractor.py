"""
Application 10: Color Palette Extractor
=======================================
Extract dominant colors from images for design/branding.

Techniques Used:
- K-Means clustering
- Color quantization
- Color space analysis
- Histogram analysis

Official Docs:
- https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class ColorPaletteExtractor:
    """
    Extract dominant colors from images using K-Means clustering.
    """

    def __init__(self, n_colors=5):
        self.n_colors = n_colors
        self.colors = None
        self.percentages = None

    def extract_colors(self, image, n_colors=None):
        """
        Extract dominant colors using K-Means.
        Returns: list of (color_bgr, percentage) tuples
        """
        if n_colors is None:
            n_colors = self.n_colors

        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)

        # K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

        # Get cluster sizes (percentages)
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100

        # Sort by percentage (most dominant first)
        sorted_idx = np.argsort(percentages)[::-1]

        self.colors = centers[sorted_idx].astype(int)
        self.percentages = percentages[sorted_idx]

        return list(zip(self.colors, self.percentages))

    def create_palette_image(self, width=400, height=100):
        """
        Create a visual palette image.
        """
        if self.colors is None:
            return None

        palette = np.zeros((height, width, 3), dtype=np.uint8)

        x = 0
        for color, pct in zip(self.colors, self.percentages):
            w = int(width * pct / 100)
            palette[:, x:x+w] = color
            x += w

        # Fill any remaining pixels
        if x < width:
            palette[:, x:] = self.colors[-1]

        return palette

    def create_swatch_image(self, swatch_size=80, padding=10):
        """
        Create color swatches with hex codes.
        """
        if self.colors is None:
            return None

        n = len(self.colors)
        width = n * (swatch_size + padding) + padding
        height = swatch_size + 60

        swatches = np.ones((height, width, 3), dtype=np.uint8) * 255

        for i, (color, pct) in enumerate(zip(self.colors, self.percentages)):
            x = padding + i * (swatch_size + padding)

            # Draw swatch
            cv2.rectangle(swatches, (x, padding), (x + swatch_size, padding + swatch_size),
                         color.tolist(), -1)
            cv2.rectangle(swatches, (x, padding), (x + swatch_size, padding + swatch_size),
                         (0, 0, 0), 1)

            # Add hex code
            hex_code = '#{:02X}{:02X}{:02X}'.format(color[2], color[1], color[0])
            cv2.putText(swatches, hex_code, (x, height - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            # Add percentage
            cv2.putText(swatches, f'{pct:.1f}%', (x + 15, height - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        return swatches

    def get_complementary(self, color):
        """
        Get complementary color.
        """
        return [255 - c for c in color]

    def get_analogous(self, color, shift=30):
        """
        Get analogous colors (nearby on color wheel).
        """
        # Convert to HSV
        color_bgr = np.uint8([[color]])
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0, 0]

        # Shift hue
        analogous = []
        for h_shift in [-shift, shift]:
            new_hsv = hsv.copy()
            new_hsv[0] = (hsv[0] + h_shift) % 180
            new_bgr = cv2.cvtColor(np.uint8([[new_hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
            analogous.append(new_bgr.tolist())

        return analogous

    def quantize_image(self, image, n_colors=None):
        """
        Reduce image to n_colors using extracted palette.
        """
        if n_colors is None:
            n_colors = self.n_colors

        # Extract colors if not done
        if self.colors is None:
            self.extract_colors(image, n_colors)

        # Reshape
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Find nearest color for each pixel
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10,
                                        cv2.KMEANS_RANDOM_CENTERS)

        # Replace pixels with cluster centers
        quantized = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)

        return quantized


def rgb_to_hex(bgr):
    """Convert BGR to hex string."""
    return '#{:02X}{:02X}{:02X}'.format(bgr[2], bgr[1], bgr[0])


def load_demo_image():
    """
    Load a colorful image for palette extraction.
    """
    for sample in ["fruits.jpg", "lena.jpg", "baboon.jpg", "building.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Create colorful placeholder
    print("No sample image found. Using synthetic image.")
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Add colorful regions
    cv2.rectangle(img, (0, 0), (200, 150), (255, 100, 50), -1)
    cv2.rectangle(img, (200, 0), (400, 150), (50, 200, 100), -1)
    cv2.rectangle(img, (0, 150), (200, 300), (100, 50, 200), -1)
    cv2.rectangle(img, (200, 150), (400, 300), (200, 200, 50), -1)

    return img


def interactive_extractor():
    """
    Interactive color palette extractor.
    """
    print("\n=== Color Palette Extractor ===")
    print("Controls:")
    print("  '3'-'9' - Set number of colors")
    print("  'q' - Quantize image")
    print("  'c' - Show complementary colors")
    print("  's' - Save palette")
    print("  'r' - Reset")
    print("  'ESC' - Quit")
    print("===============================\n")

    original = load_demo_image()
    extractor = ColorPaletteExtractor(n_colors=5)

    # Initial extraction
    colors = extractor.extract_colors(original)
    print("\nDominant colors:")
    for color, pct in colors:
        print(f"  {rgb_to_hex(color)}: {pct:.1f}%")

    while True:
        # Create display
        palette = extractor.create_palette_image(original.shape[1], 50)
        swatches = extractor.create_swatch_image()

        # Stack displays
        img_with_palette = np.vstack([original, palette])

        cv2.imshow("Image with Palette", img_with_palette)
        cv2.imshow("Color Swatches", swatches)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif ord('3') <= key <= ord('9'):
            n = key - ord('0')
            extractor.n_colors = n
            colors = extractor.extract_colors(original, n)
            print(f"\nExtracted {n} colors:")
            for color, pct in colors:
                print(f"  {rgb_to_hex(color)}: {pct:.1f}%")
        elif key == ord('q'):
            quantized = extractor.quantize_image(original)
            cv2.imshow("Quantized Image", quantized)
        elif key == ord('c'):
            if extractor.colors is not None:
                print("\nComplementary colors:")
                for color in extractor.colors:
                    comp = extractor.get_complementary(color)
                    print(f"  {rgb_to_hex(color)} -> {rgb_to_hex(comp)}")
        elif key == ord('s'):
            cv2.imwrite("color_palette.png", swatches)
            print("Saved: color_palette.png")
        elif key == ord('r'):
            colors = extractor.extract_colors(original)

    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo showing color extraction.
    """
    print("\n=== Color Palette Extractor Demo ===\n")

    original = load_demo_image()
    extractor = ColorPaletteExtractor(n_colors=5)

    # Extract colors
    colors = extractor.extract_colors(original)

    print("Dominant colors (BGR -> Hex):")
    for i, (color, pct) in enumerate(colors, 1):
        hex_code = rgb_to_hex(color)
        print(f"  {i}. {hex_code} - {pct:.1f}%")

    # Create visualizations
    palette = extractor.create_palette_image(original.shape[1], 60)
    swatches = extractor.create_swatch_image(swatch_size=100, padding=15)
    quantized = extractor.quantize_image(original)

    # Stack original and palette
    combined = np.vstack([original, palette])

    # Add title
    cv2.putText(combined, "Original Image + Extracted Palette", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Color Palette Extraction", combined)
    cv2.imshow("Color Swatches", swatches)
    cv2.imshow("Quantized Image", quantized)

    print("\nApplications:")
    print("- Design color schemes")
    print("- Brand color extraction")
    print("- Image compression (color quantization)")
    print("- Style transfer preparation")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 10: Color Palette Extractor")
    print("=" * 60)

    try:
        interactive_extractor()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
