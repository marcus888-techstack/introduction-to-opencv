"""
Sample Data Module
==================
Provides easy access to sample images and videos for tutorials.

Usage:
    from sample_data import get_image, get_video, SAMPLE_DIR

    # Load a sample image
    img = get_image("lena.jpg")

    # Get path to video
    video_path = get_video("vtest.avi")
"""

import os
import cv2
import numpy as np

# Directory containing sample files
SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_sample_path(filename):
    """
    Get full path to a sample file, downloading if necessary.
    """
    from .download_samples import get_sample_path as _get_path
    return _get_path(filename)


def get_image(filename, flags=cv2.IMREAD_COLOR):
    """
    Load a sample image.

    Args:
        filename: Name of the sample image (e.g., "lena.jpg")
        flags: cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, etc.

    Returns:
        numpy array of the image, or None if not found

    Example:
        img = get_image("lena.jpg")
        gray = get_image("lena.jpg", cv2.IMREAD_GRAYSCALE)
    """
    filepath = get_sample_path(filename)

    if not os.path.exists(filepath):
        print(f"Warning: Sample image not found: {filename}")
        print("Run: python -m sample_data.download_samples")
        return None

    img = cv2.imread(filepath, flags)

    if img is None:
        print(f"Warning: Could not load image: {filename}")
        return None

    return img


def get_video(filename):
    """
    Get path to a sample video file.

    Args:
        filename: Name of the sample video (e.g., "vtest.avi")

    Returns:
        Full path to the video file

    Example:
        cap = cv2.VideoCapture(get_video("vtest.avi"))
    """
    return get_sample_path(filename)


def get_image_or_camera(filename=None, camera_id=0):
    """
    Get image from file or camera.

    If filename provided and exists, load it.
    Otherwise, capture from camera.

    Args:
        filename: Optional sample image name
        camera_id: Camera ID for fallback

    Returns:
        (image, source) tuple where source is "file" or "camera"
    """
    if filename:
        img = get_image(filename)
        if img is not None:
            return img, "file"

    # Try camera
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame, "camera"

    return None, None


def list_samples():
    """List all available sample files."""
    from .download_samples import SAMPLE_FILES, ALT_SOURCES

    print("\nAvailable samples:")
    print("-" * 40)

    images = [f for f in SAMPLE_FILES.keys() if not f.endswith(('.avi', '.mp4'))]
    videos = [f for f in SAMPLE_FILES.keys() if f.endswith(('.avi', '.mp4'))]

    print("\nImages:")
    for f in sorted(images):
        print(f"  {f}")

    print("\nVideos:")
    for f in sorted(videos):
        print(f"  {f}")


# Common sample images for quick access
SAMPLES = {
    "face": "lena.jpg",
    "fruits": "fruits.jpg",
    "building": "building.jpg",
    "text": "imageTextN.png",
    "sudoku": "sudoku.png",
    "chessboard": "chessboard.png",
    "box": "box.png",
    "scene": "box_in_scene.png",
}


def quick_load(name):
    """
    Quick load common sample images by category.

    Args:
        name: "face", "fruits", "building", "text", "sudoku",
              "chessboard", "box", "scene"

    Example:
        face_img = quick_load("face")
    """
    if name in SAMPLES:
        return get_image(SAMPLES[name])
    else:
        print(f"Unknown sample: {name}")
        print(f"Available: {list(SAMPLES.keys())}")
        return None
