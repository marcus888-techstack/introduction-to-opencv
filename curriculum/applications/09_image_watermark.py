"""
Application 09: Image Watermarking
==================================
Add visible and invisible watermarks to images.

Techniques Used:
- Image blending (addWeighted)
- Alpha channel manipulation
- Bitwise operations
- DCT-based watermarking (invisible)

Official Docs:
- https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class Watermarker:
    """
    Image watermarking with various methods.
    """

    @staticmethod
    def text_watermark(image, text, position='bottom-right', opacity=0.5,
                       font_scale=1.0, color=(255, 255, 255)):
        """
        Add visible text watermark.
        """
        result = image.copy()
        height, width = image.shape[:2]

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate position
        padding = 20
        if position == 'bottom-right':
            x = width - text_w - padding
            y = height - padding
        elif position == 'bottom-left':
            x = padding
            y = height - padding
        elif position == 'top-right':
            x = width - text_w - padding
            y = text_h + padding
        elif position == 'top-left':
            x = padding
            y = text_h + padding
        elif position == 'center':
            x = (width - text_w) // 2
            y = (height + text_h) // 2
        else:
            x, y = position

        # Create overlay
        overlay = result.copy()
        cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)

        # Blend
        result = cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0)

        return result

    @staticmethod
    def image_watermark(image, watermark, position='bottom-right', opacity=0.3, scale=0.2):
        """
        Add visible image watermark (logo).
        """
        result = image.copy()
        h, w = image.shape[:2]
        wh, ww = watermark.shape[:2]

        # Scale watermark
        new_ww = int(w * scale)
        new_wh = int(wh * (new_ww / ww))
        watermark_resized = cv2.resize(watermark, (new_ww, new_wh))

        # Calculate position
        padding = 20
        if position == 'bottom-right':
            x = w - new_ww - padding
            y = h - new_wh - padding
        elif position == 'bottom-left':
            x = padding
            y = h - new_wh - padding
        elif position == 'top-right':
            x = w - new_ww - padding
            y = padding
        elif position == 'top-left':
            x = padding
            y = padding
        elif position == 'center':
            x = (w - new_ww) // 2
            y = (h - new_wh) // 2
        else:
            x, y = position

        # Ensure watermark fits
        x = max(0, min(x, w - new_ww))
        y = max(0, min(y, h - new_wh))

        # Extract ROI
        roi = result[y:y+new_wh, x:x+new_ww]

        # Handle alpha channel if present
        if watermark_resized.shape[2] == 4:
            # BGRA watermark
            alpha = watermark_resized[:, :, 3] / 255.0 * opacity
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * watermark_resized[:, :, c]
        else:
            # BGR watermark - simple blend
            blended = cv2.addWeighted(watermark_resized, opacity, roi, 1 - opacity, 0)
            result[y:y+new_wh, x:x+new_ww] = blended

        return result

    @staticmethod
    def tiled_watermark(image, text, opacity=0.1, angle=-30, spacing=150):
        """
        Add tiled/repeated watermark across entire image.
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Create transparent overlay
        overlay = np.zeros_like(image)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (200, 200, 200)

        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw tiled text
        for y_pos in range(-h, 2*h, spacing):
            for x_pos in range(-w, 2*w, spacing):
                # Rotate text position
                cx, cy = w // 2, h // 2
                rad = np.radians(angle)
                rx = int(cx + (x_pos - cx) * np.cos(rad) - (y_pos - cy) * np.sin(rad))
                ry = int(cy + (x_pos - cx) * np.sin(rad) + (y_pos - cy) * np.cos(rad))

                if 0 <= rx < w and 0 <= ry < h:
                    cv2.putText(overlay, text, (rx, ry), font, font_scale, color, thickness)

        # Blend
        result = cv2.addWeighted(overlay, opacity, result, 1, 0)

        return result

    @staticmethod
    def invisible_watermark(image, watermark_text, strength=10):
        """
        Add invisible watermark using LSB (Least Significant Bit).
        Simple implementation for demonstration.
        """
        result = image.copy()

        # Convert text to binary
        binary_text = ''.join(format(ord(c), '08b') for c in watermark_text)
        binary_text += '00000000'  # Null terminator

        # Embed in blue channel LSB
        flat = result[:, :, 0].flatten()

        for i, bit in enumerate(binary_text):
            if i >= len(flat):
                break
            # Clear LSB and set new bit
            flat[i] = (flat[i] & 0xFE) | int(bit)

        result[:, :, 0] = flat.reshape(result[:, :, 0].shape)

        return result

    @staticmethod
    def extract_invisible_watermark(image, max_length=100):
        """
        Extract invisible watermark from LSB.
        """
        flat = image[:, :, 0].flatten()

        # Extract bits
        bits = ''
        for i in range(min(max_length * 8, len(flat))):
            bits += str(flat[i] & 1)

        # Convert to text
        text = ''
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if byte == '00000000':
                break
            text += chr(int(byte, 2))

        return text


def create_logo():
    """
    Create a simple logo for watermarking demo.
    """
    # Create logo with alpha channel
    logo = np.zeros((100, 200, 4), dtype=np.uint8)

    # Draw circle
    cv2.circle(logo, (50, 50), 40, (255, 200, 0, 255), -1)

    # Draw text
    cv2.putText(logo, "OpenCV", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)

    return logo


def load_demo_image():
    """
    Load a real image for watermarking demo.
    """
    for sample in ["lena.jpg", "fruits.jpg", "baboon.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Create placeholder
    print("No sample image found. Using synthetic image.")
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    cv2.putText(img, "Sample Image", (150, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
    return img


def interactive_watermark():
    """
    Interactive watermarking tool.
    """
    print("\n=== Image Watermarking Tool ===")
    print("Controls:")
    print("  '1' - Text watermark")
    print("  '2' - Logo watermark")
    print("  '3' - Tiled watermark")
    print("  '4' - Invisible watermark")
    print("  '5' - Extract invisible watermark")
    print("  'p' - Change position")
    print("  '+'/'-' - Adjust opacity")
    print("  'r' - Reset image")
    print("  's' - Save result")
    print("  'q' - Quit")
    print("===============================\n")

    # Load image
    original = load_demo_image()
    result = original.copy()
    logo = create_logo()

    watermarker = Watermarker()
    positions = ['bottom-right', 'bottom-left', 'top-right', 'top-left', 'center']
    pos_idx = 0
    opacity = 0.5

    while True:
        # Display
        display = result.copy()
        cv2.putText(display, f"Position: {positions[pos_idx]} | Opacity: {opacity:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Watermarking Tool", display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            result = watermarker.text_watermark(original, "COPYRIGHT 2024",
                                                positions[pos_idx], opacity)
            print("Applied text watermark")
        elif key == ord('2'):
            result = watermarker.image_watermark(original, logo,
                                                 positions[pos_idx], opacity)
            print("Applied logo watermark")
        elif key == ord('3'):
            result = watermarker.tiled_watermark(original, "SAMPLE", opacity * 0.3)
            print("Applied tiled watermark")
        elif key == ord('4'):
            result = watermarker.invisible_watermark(original, "SECRET_MARK_2024")
            print("Applied invisible watermark (LSB)")
        elif key == ord('5'):
            extracted = watermarker.extract_invisible_watermark(result)
            print(f"Extracted watermark: '{extracted}'")
        elif key == ord('p'):
            pos_idx = (pos_idx + 1) % len(positions)
            print(f"Position: {positions[pos_idx]}")
        elif key == ord('+') or key == ord('='):
            opacity = min(1.0, opacity + 0.1)
        elif key == ord('-'):
            opacity = max(0.1, opacity - 0.1)
        elif key == ord('r'):
            result = original.copy()
            print("Reset to original")
        elif key == ord('s'):
            cv2.imwrite("watermarked_image.jpg", result)
            print("Saved: watermarked_image.jpg")

    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo showing all watermark types.
    """
    print("\n=== Watermarking Demo ===\n")

    original = load_demo_image()
    logo = create_logo()
    watermarker = Watermarker()

    # Apply different watermarks
    text_wm = watermarker.text_watermark(original, "COPYRIGHT 2024", opacity=0.7)
    logo_wm = watermarker.image_watermark(original, logo, opacity=0.5)
    tiled_wm = watermarker.tiled_watermark(original, "DRAFT", opacity=0.15)
    invisible_wm = watermarker.invisible_watermark(original, "SECRET_2024")

    # Extract invisible watermark
    extracted = watermarker.extract_invisible_watermark(invisible_wm)
    print(f"Invisible watermark extracted: '{extracted}'")

    # Create display
    row1 = np.hstack([original, text_wm])
    row2 = np.hstack([logo_wm, tiled_wm])
    display = np.vstack([row1, row2])
    display = cv2.resize(display, (800, 600))

    # Add labels
    cv2.putText(display, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display, "Text Watermark", (410, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display, "Logo Watermark", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display, "Tiled Watermark", (410, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Watermarking Demo", display)

    print("\nWatermarking types:")
    print("1. Text watermark - Simple text overlay")
    print("2. Logo watermark - Image overlay with alpha")
    print("3. Tiled watermark - Repeated pattern")
    print("4. Invisible watermark - LSB steganography")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 09: Image Watermarking")
    print("=" * 60)

    try:
        interactive_watermark()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
