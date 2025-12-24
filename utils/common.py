"""
Common Utility Functions for OpenCV Projects
=============================================
Shared functions used across multiple projects.
"""

import cv2
import numpy as np
from pathlib import Path


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        inter: Interpolation method

    Returns:
        Resized image
    """
    dim = None
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def draw_text_with_background(image, text, position, font_scale=0.7,
                               color=(255, 255, 255), bg_color=(0, 0, 0),
                               thickness=2, padding=5):
    """
    Draw text with a background rectangle for better visibility.

    Args:
        image: Image to draw on
        text: Text to display
        position: (x, y) position for text
        font_scale: Font size
        color: Text color (BGR)
        bg_color: Background color (BGR)
        thickness: Text thickness
        padding: Padding around text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position

    # Draw background rectangle
    cv2.rectangle(image,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + baseline + padding),
                  bg_color, -1)

    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

    return image


def stack_images(images, cols=2, resize_width=400):
    """
    Stack multiple images in a grid for display.

    Args:
        images: List of images
        cols: Number of columns
        resize_width: Width to resize each image

    Returns:
        Stacked image
    """
    # Resize all images
    resized = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized.append(resize_with_aspect_ratio(img, width=resize_width))

    # Pad to make same height
    max_h = max(img.shape[0] for img in resized)
    padded = []
    for img in resized:
        h, w = img.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    # Stack in rows
    rows = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i+cols]
        # Pad if not enough images
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))

    return np.vstack(rows)


def create_video_writer(output_path, frame_size, fps=30):
    """
    Create a video writer with common settings.

    Args:
        output_path: Output file path
        frame_size: (width, height) tuple
        fps: Frames per second

    Returns:
        cv2.VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)


def get_color_by_id(obj_id, palette=None):
    """
    Get a consistent color for an object ID.

    Args:
        obj_id: Object identifier
        palette: Optional color palette

    Returns:
        BGR color tuple
    """
    if palette is None:
        palette = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]

    return palette[obj_id % len(palette)]


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two boxes.

    Args:
        box1: (x, y, w, h) tuple
        box2: (x, y, w, h) tuple

    Returns:
        IoU value (0-1)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Intersection coordinates
    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1 + w1, x2 + w2) - xi
    hi = min(y1 + h1, y2 + h2) - yi

    if wi <= 0 or hi <= 0:
        return 0.0

    intersection = wi * hi
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union if union > 0 else 0.0


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


class FPSCounter:
    """Simple FPS counter for video processing."""

    def __init__(self, avg_frames=30):
        self.times = []
        self.avg_frames = avg_frames

    def tick(self):
        """Record current time."""
        import time
        self.times.append(time.time())
        if len(self.times) > self.avg_frames:
            self.times.pop(0)

    def get_fps(self):
        """Calculate average FPS."""
        if len(self.times) < 2:
            return 0.0
        return len(self.times) / (self.times[-1] - self.times[0])


# Example usage
if __name__ == "__main__":
    print("Common Utilities for OpenCV Projects")
    print("=" * 40)
    print()
    print("Available functions:")
    print("  - resize_with_aspect_ratio()")
    print("  - draw_text_with_background()")
    print("  - stack_images()")
    print("  - create_video_writer()")
    print("  - get_color_by_id()")
    print("  - compute_iou()")
    print("  - ensure_dir()")
    print()
    print("Classes:")
    print("  - FPSCounter")
