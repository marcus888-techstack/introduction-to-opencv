"""
Application 13: Virtual Background
=================================
Replace background in video using segmentation techniques.

Techniques Used:
- Background subtraction
- GrabCut segmentation
- Color-based segmentation
- Image blending

Official Docs:
- https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html
- https://docs.opencv.org/4.x/db/d5c/tutorial_py_bg_subtraction.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class VirtualBackground:
    """
    Virtual background replacement using various techniques.
    """

    def __init__(self):
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # KNN subtractor (alternative)
        self.knn_subtractor = cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400, detectShadows=True
        )

        # Current method
        self.method = 'color'  # 'color', 'grabcut', 'bg_sub'

        # Color range for green screen (HSV)
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([85, 255, 255])

        # Smoothing
        self.prev_mask = None
        self.smoothing = 0.5

    def color_key(self, frame, background):
        """
        Green screen / color keying method.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for green color
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Invert mask (foreground is non-green)
        mask = cv2.bitwise_not(mask)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Blur edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return self._blend(frame, background, mask)

    def bg_subtraction(self, frame, background):
        """
        Background subtraction method.
        """
        # Get foreground mask
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (shadows are gray, foreground is white)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Smooth with previous mask
        if self.prev_mask is not None:
            fg_mask = cv2.addWeighted(fg_mask, 1 - self.smoothing,
                                       self.prev_mask, self.smoothing, 0)
        self.prev_mask = fg_mask.copy()

        # Blur edges
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        return self._blend(frame, background, fg_mask)

    def grabcut_segment(self, frame, background, rect=None):
        """
        GrabCut segmentation method.
        Note: Slow, best for static images.
        """
        h, w = frame.shape[:2]

        # Default rectangle (assume person in center)
        if rect is None:
            rect = (int(w * 0.1), int(h * 0.05), int(w * 0.8), int(h * 0.95))

        # Initialize mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # Run GrabCut
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Create binary mask (0,2 = background, 1,3 = foreground)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        return self._blend(frame, background, fg_mask)

    def _blend(self, foreground, background, mask):
        """
        Blend foreground and background using mask.
        """
        # Resize background to match foreground
        h, w = foreground.shape[:2]
        background = cv2.resize(background, (w, h))

        # Convert mask to 3 channels and normalize
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255

        # Blend
        result = foreground.astype(float) * mask_3ch + background.astype(float) * (1 - mask_3ch)

        return result.astype(np.uint8)

    def process(self, frame, background):
        """
        Process frame with current method.
        """
        if self.method == 'color':
            return self.color_key(frame, background)
        elif self.method == 'bg_sub':
            return self.bg_subtraction(frame, background)
        elif self.method == 'grabcut':
            return self.grabcut_segment(frame, background)
        else:
            return frame


def load_background_images():
    """
    Load background images for virtual background.
    """
    backgrounds = []

    # Try real background images
    samples = ["beach.jpg", "office.jpg", "space.jpg", "nature.jpg",
               "mountain.jpg", "city.jpg", "room.jpg"]

    for sample in samples:
        img = get_image(sample)
        if img is not None:
            backgrounds.append((sample, img))

    # Create synthetic backgrounds if needed
    if len(backgrounds) < 3:
        # Gradient background
        gradient = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            gradient[i, :] = (
                int(50 + (i / 480) * 150),
                int(100 + (i / 480) * 100),
                int(200 - (i / 480) * 100)
            )
        backgrounds.append(("Gradient", gradient))

        # Solid color
        solid = np.ones((480, 640, 3), dtype=np.uint8)
        solid[:] = (40, 40, 40)
        backgrounds.append(("Dark Gray", solid))

        # Blurred office
        blur_bg = np.ones((480, 640, 3), dtype=np.uint8) * 180
        cv2.rectangle(blur_bg, (50, 100), (200, 400), (150, 150, 150), -1)
        cv2.rectangle(blur_bg, (250, 50), (400, 200), (120, 120, 120), -1)
        blur_bg = cv2.GaussianBlur(blur_bg, (51, 51), 0)
        backgrounds.append(("Blurred Room", blur_bg))

    return backgrounds


def load_demo_frame():
    """
    Load a demo frame for static testing.
    """
    # Try to load person image
    for sample in ["person.jpg", "portrait.jpg", "face.jpg", "lena.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return img

    # Create synthetic person placeholder
    print("No person sample found. Using synthetic frame.")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Green screen background
    frame[:] = (0, 255, 0)

    # Person silhouette (gray)
    cv2.ellipse(frame, (320, 150), (60, 80), 0, 0, 360, (150, 150, 150), -1)  # Head
    cv2.rectangle(frame, (240, 230), (400, 480), (150, 150, 150), -1)  # Body

    cv2.putText(frame, "Green Screen Demo", (180, 450),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame


def interactive_background():
    """
    Interactive virtual background with webcam.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    print("\n=== Virtual Background ===")
    print("Controls:")
    print("  '1' - Color keying (green screen)")
    print("  '2' - Background subtraction")
    print("  '3' - GrabCut (slow, press once)")
    print("  'n' - Next background")
    print("  'p' - Previous background")
    print("  '+'/'-' - Adjust color range")
    print("  's' - Save screenshot")
    print("  'r' - Reset background model")
    print("  'q' - Quit")
    print("==========================\n")

    vb = VirtualBackground()
    backgrounds = load_background_images()
    bg_idx = 0

    print(f"Loaded {len(backgrounds)} backgrounds")
    print("For best results with 'color' mode, use a green screen behind you.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Get current background
        bg_name, background = backgrounds[bg_idx]

        # Process frame
        result = vb.process(frame, background)

        # Display info
        cv2.putText(result, f"Method: {vb.method} | BG: {bg_name}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Virtual Background", result)
        cv2.imshow("Original", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            vb.method = 'color'
            print("Switched to color keying")
        elif key == ord('2'):
            vb.method = 'bg_sub'
            print("Switched to background subtraction")
        elif key == ord('3'):
            vb.method = 'grabcut'
            print("Running GrabCut (this will be slow)...")
        elif key == ord('n'):
            bg_idx = (bg_idx + 1) % len(backgrounds)
            print(f"Background: {backgrounds[bg_idx][0]}")
        elif key == ord('p'):
            bg_idx = (bg_idx - 1) % len(backgrounds)
            print(f"Background: {backgrounds[bg_idx][0]}")
        elif key == ord('+') or key == ord('='):
            vb.lower_green[0] = max(0, vb.lower_green[0] - 5)
            vb.upper_green[0] = min(180, vb.upper_green[0] + 5)
            print(f"Green range: {vb.lower_green[0]}-{vb.upper_green[0]}")
        elif key == ord('-'):
            vb.lower_green[0] = min(180, vb.lower_green[0] + 5)
            vb.upper_green[0] = max(0, vb.upper_green[0] - 5)
        elif key == ord('r'):
            vb.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            vb.prev_mask = None
            print("Reset background model")
        elif key == ord('s'):
            cv2.imwrite("virtual_background.jpg", result)
            print("Saved: virtual_background.jpg")

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with static image.
    """
    print("\n=== Virtual Background Demo ===\n")

    # Load frame and backgrounds
    frame = load_demo_frame()
    backgrounds = load_background_images()

    vb = VirtualBackground()

    # Create results for each background
    results = []
    for bg_name, background in backgrounds[:4]:  # Limit to 4 for display
        result = vb.color_key(frame, background)
        cv2.putText(result, bg_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        results.append(result)

    # Create grid display
    if len(results) >= 4:
        row1 = np.hstack([frame, results[0]])
        row2 = np.hstack([results[1], results[2]])
        display = np.vstack([row1, row2])
    else:
        display = np.hstack([frame] + results[:2])

    display = cv2.resize(display, (800, 600))

    cv2.imshow("Virtual Background Demo", display)

    print("Methods available:")
    print("1. Color keying - Requires green/blue screen")
    print("2. Background subtraction - Learns background over time")
    print("3. GrabCut - Slow but works without setup")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 13: Virtual Background")
    print("=" * 60)

    try:
        interactive_background()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
