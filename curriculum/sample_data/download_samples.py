"""
Sample Data Downloader
======================
Downloads real sample images and videos for OpenCV tutorials.

Sources:
- OpenCV GitHub samples
- Public domain images

Usage:
    python download_samples.py           # Download all samples
    python download_samples.py --check   # Check which samples exist
"""

import os
import urllib.request
import sys

# Sample data directory
SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sample files to download from OpenCV GitHub
OPENCV_SAMPLES_BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"

SAMPLE_FILES = {
    # Basic images
    "lena.jpg": OPENCV_SAMPLES_BASE + "lena.jpg",
    "fruits.jpg": OPENCV_SAMPLES_BASE + "fruits.jpg",
    "baboon.jpg": OPENCV_SAMPLES_BASE + "baboon.jpg",
    "building.jpg": OPENCV_SAMPLES_BASE + "building.jpg",
    "starry_night.jpg": OPENCV_SAMPLES_BASE + "starry_night.jpg",
    "butterfly.jpg": OPENCV_SAMPLES_BASE + "butterfly.jpg",

    # Face detection
    "lena_face.jpg": OPENCV_SAMPLES_BASE + "lena.jpg",
    "messi5.jpg": OPENCV_SAMPLES_BASE + "messi5.jpg",

    # Feature matching
    "box.png": OPENCV_SAMPLES_BASE + "box.png",
    "box_in_scene.png": OPENCV_SAMPLES_BASE + "box_in_scene.png",
    "blox.jpg": OPENCV_SAMPLES_BASE + "blox.jpg",

    # Template matching
    "lena_tmpl.jpg": OPENCV_SAMPLES_BASE + "lena_tmpl.jpg",

    # Calibration
    "left01.jpg": OPENCV_SAMPLES_BASE + "left01.jpg",
    "left02.jpg": OPENCV_SAMPLES_BASE + "left02.jpg",
    "right01.jpg": OPENCV_SAMPLES_BASE + "right01.jpg",
    "right02.jpg": OPENCV_SAMPLES_BASE + "right02.jpg",
    "chessboard.png": OPENCV_SAMPLES_BASE + "chessboard.png",

    # Text/Document/OCR
    "imageTextN.png": OPENCV_SAMPLES_BASE + "imageTextN.png",
    "imageTextR.png": OPENCV_SAMPLES_BASE + "imageTextR.png",
    "licenseplate_motion.jpg": OPENCV_SAMPLES_BASE + "licenseplate_motion.jpg",
    "text_defocus.jpg": OPENCV_SAMPLES_BASE + "text_defocus.jpg",

    # Morphology (note: j.png may not be available in newer OpenCV versions)

    # Shapes and contours
    "pic1.png": OPENCV_SAMPLES_BASE + "pic1.png",
    "pic2.png": OPENCV_SAMPLES_BASE + "pic2.png",
    "pic3.png": OPENCV_SAMPLES_BASE + "pic3.png",
    "pic4.png": OPENCV_SAMPLES_BASE + "pic4.png",
    "pic5.png": OPENCV_SAMPLES_BASE + "pic5.png",
    "pic6.png": OPENCV_SAMPLES_BASE + "pic6.png",

    # Gradient/edges
    "sudoku.png": OPENCV_SAMPLES_BASE + "sudoku.png",

    # Histogram
    "home.jpg": OPENCV_SAMPLES_BASE + "home.jpg",

    # Image stitching / Panorama
    "Blender_Suzanne1.jpg": OPENCV_SAMPLES_BASE + "Blender_Suzanne1.jpg",
    "Blender_Suzanne2.jpg": OPENCV_SAMPLES_BASE + "Blender_Suzanne2.jpg",
    "boat1.jpg": OPENCV_SAMPLES_BASE + "boat1.jpg",
    "boat2.jpg": OPENCV_SAMPLES_BASE + "boat2.jpg",
    "boat3.jpg": OPENCV_SAMPLES_BASE + "boat3.jpg",
    "boat4.jpg": OPENCV_SAMPLES_BASE + "boat4.jpg",
    "boat5.jpg": OPENCV_SAMPLES_BASE + "boat5.jpg",
    "boat6.jpg": OPENCV_SAMPLES_BASE + "boat6.jpg",

    # Machine Learning
    "digits.png": OPENCV_SAMPLES_BASE + "digits.png",  # 5000 handwritten digits (20x20 each)

    # Videos
    "vtest.avi": OPENCV_SAMPLES_BASE + "vtest.avi",
}

# Alternative sources for common test images (if OpenCV samples are unavailable)
ALT_SOURCES = {
    # Additional images can be added here if needed
}

# Multi-View / SFM Images
# ETH3D Dataset preview images (outdoor scenes for feature matching & SFM)
# https://www.eth3d.net/datasets
ETH3D_BASE = "https://www.eth3d.net/img/"

MULTIVIEW_IMAGES = {
    # Building facade - excellent for feature matching and SFM
    "facade_view1.jpg": ETH3D_BASE + "dslr_facade.jpg",

    # Courtyard - outdoor scene with good features
    "courtyard_view1.jpg": ETH3D_BASE + "dslr_courtyard.jpg",

    # Terrace - outdoor with depth variation
    "terrace_view1.jpg": ETH3D_BASE + "dslr_terrace.jpg",

    # Playground - outdoor with structures
    "playground_view1.jpg": ETH3D_BASE + "dslr_playground.jpg",

    # Relief sculpture - good texture for matching
    "relief_view1.jpg": ETH3D_BASE + "dslr_relief.jpg",
}

# Middlebury Stereo Benchmark Datasets
# https://vision.middlebury.edu/stereo/data/
MIDDLEBURY_BASE = "https://vision.middlebury.edu/stereo/data/"

STEREO_DATASETS = {
    # Tsukuba (2001) - Classic stereo benchmark, 384x288
    "tsukuba_left.ppm": MIDDLEBURY_BASE + "scenes2001/data/tsukuba/scene1.row3.col1.ppm",
    "tsukuba_right.ppm": MIDDLEBURY_BASE + "scenes2001/data/tsukuba/scene1.row3.col3.ppm",
    "tsukuba_disp.pgm": MIDDLEBURY_BASE + "scenes2001/data/tsukuba/truedisp.row3.col3.pgm",

    # Cones (2003) - 450x375, complex geometry
    "cones_left.png": MIDDLEBURY_BASE + "scenes2003/newdata/cones/im2.png",
    "cones_right.png": MIDDLEBURY_BASE + "scenes2003/newdata/cones/im6.png",
    "cones_disp.png": MIDDLEBURY_BASE + "scenes2003/newdata/cones/disp2.png",

    # Teddy (2003) - 450x375, teddy bear scene
    "teddy_left.png": MIDDLEBURY_BASE + "scenes2003/newdata/teddy/im2.png",
    "teddy_right.png": MIDDLEBURY_BASE + "scenes2003/newdata/teddy/im6.png",
    "teddy_disp.png": MIDDLEBURY_BASE + "scenes2003/newdata/teddy/disp2.png",
}


# =============================================================================
# MOT / MCMOT - Person Detection & Re-ID Models
# =============================================================================

# YOLOv4-tiny for person detection
# https://github.com/AlexeyAB/darknet
YOLO_MODELS = {
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

# Person Re-ID Model from OpenCV Zoo
# https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtureid
REID_MODELS = {
    "person_reid_youtu_2021nov.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/person_reid_youtureid/person_reid_youtu_2021nov.onnx",
}

# Test videos for MOT/MCMOT (using OpenCV samples)
MOT_VIDEOS = {
    # OpenCV pedestrian video - good for tracking demos
    "vtest.avi": OPENCV_SAMPLES_BASE + "vtest.avi",
}


# =============================================================================
# Machine Learning - Additional Samples
# =============================================================================

ML_SAMPLES = {
    # Handwritten digits (included in SAMPLE_FILES as digits.png)
    # Additional ML-related images
    "aero1.jpg": OPENCV_SAMPLES_BASE + "aero1.jpg",
    "aero2.jpg": OPENCV_SAMPLES_BASE + "aero2.jpg",
    "aero3.jpg": OPENCV_SAMPLES_BASE + "aero3.jpg",
}


def download_file(url, filename):
    """Download a file from URL."""
    filepath = os.path.join(SAMPLE_DIR, filename)

    if os.path.exists(filepath):
        print(f"  [EXISTS] {filename}")
        return True

    try:
        print(f"  [DOWNLOADING] {filename}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, filepath)
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED ({e})")
        return False


def download_all():
    """Download all sample files."""
    print("=" * 50)
    print("Downloading OpenCV Sample Data")
    print("=" * 50)
    print(f"\nTarget directory: {SAMPLE_DIR}\n")

    success = 0
    failed = 0

    print("OpenCV Samples:")
    for filename, url in SAMPLE_FILES.items():
        if download_file(url, filename):
            success += 1
        else:
            failed += 1

    print("\nAlternative Sources:")
    for filename, url in ALT_SOURCES.items():
        if download_file(url, filename):
            success += 1
        else:
            failed += 1

    print("\nMulti-View / SFM Images:")
    for filename, url in MULTIVIEW_IMAGES.items():
        if download_file(url, filename):
            success += 1
        else:
            failed += 1

    print("\nMiddlebury Stereo Datasets:")
    for filename, url in STEREO_DATASETS.items():
        if download_file(url, filename):
            success += 1
        else:
            failed += 1

    print("\nYOLO Detection Models:")
    for filename, url in YOLO_MODELS.items():
        if download_file(url, filename):
            success += 1
        else:
            failed += 1

    print("\nPerson Re-ID Models:")
    for filename, url in REID_MODELS.items():
        if download_file(url, filename):
            success += 1
        else:
            failed += 1

    print("\nMOT/MCMOT Videos:")
    for filename, url in MOT_VIDEOS.items():
        if download_file(url, filename):
            success += 1
        else:
            failed += 1

    print("\nMachine Learning Samples:")
    for filename, url in ML_SAMPLES.items():
        if download_file(url, filename):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print(f"Complete: {success} downloaded, {failed} failed")
    print("=" * 50)

    return failed == 0


def check_samples():
    """Check which samples exist."""
    print("=" * 50)
    print("Sample Data Status")
    print("=" * 50)

    all_files = {**SAMPLE_FILES, **ALT_SOURCES, **MULTIVIEW_IMAGES, **STEREO_DATASETS,
                 **YOLO_MODELS, **REID_MODELS, **MOT_VIDEOS, **ML_SAMPLES}

    exists = 0
    missing = 0

    for filename in sorted(all_files.keys()):
        filepath = os.path.join(SAMPLE_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  [OK] {filename} ({size:,} bytes)")
            exists += 1
        else:
            print(f"  [MISSING] {filename}")
            missing += 1

    print("\n" + "=" * 50)
    print(f"Status: {exists} exist, {missing} missing")
    print("=" * 50)

    if missing > 0:
        print("\nRun 'python download_samples.py' to download missing files.")


def get_sample_path(filename):
    """
    Get the full path to a sample file.
    Downloads it if it doesn't exist.

    Usage:
        from sample_data.download_samples import get_sample_path
        img = cv2.imread(get_sample_path("lena.jpg"))
    """
    filepath = os.path.join(SAMPLE_DIR, filename)

    if not os.path.exists(filepath):
        # Try to download
        if filename in SAMPLE_FILES:
            download_file(SAMPLE_FILES[filename], filename)
        elif filename in ALT_SOURCES:
            download_file(ALT_SOURCES[filename], filename)
        elif filename in MULTIVIEW_IMAGES:
            download_file(MULTIVIEW_IMAGES[filename], filename)
        elif filename in STEREO_DATASETS:
            download_file(STEREO_DATASETS[filename], filename)
        elif filename in YOLO_MODELS:
            download_file(YOLO_MODELS[filename], filename)
        elif filename in REID_MODELS:
            download_file(REID_MODELS[filename], filename)
        elif filename in MOT_VIDEOS:
            download_file(MOT_VIDEOS[filename], filename)
        elif filename in ML_SAMPLES:
            download_file(ML_SAMPLES[filename], filename)

    return filepath


def list_available():
    """List all available sample files."""
    print("\nAvailable sample files:")
    print("-" * 30)

    for filename in sorted(SAMPLE_FILES.keys()):
        print(f"  - {filename}")

    print("\nAlternative sources:")
    for filename in sorted(ALT_SOURCES.keys()):
        print(f"  - {filename}")

    print("\nMulti-View / SFM Images:")
    for filename in sorted(MULTIVIEW_IMAGES.keys()):
        print(f"  - {filename}")

    print("\nMiddlebury Stereo Datasets:")
    for filename in sorted(STEREO_DATASETS.keys()):
        print(f"  - {filename}")

    print("\nYOLO Detection Models:")
    for filename in sorted(YOLO_MODELS.keys()):
        print(f"  - {filename}")

    print("\nPerson Re-ID Models:")
    for filename in sorted(REID_MODELS.keys()):
        print(f"  - {filename}")

    print("\nMOT/MCMOT Videos:")
    for filename in sorted(MOT_VIDEOS.keys()):
        print(f"  - {filename}")

    print("\nMachine Learning Samples:")
    for filename in sorted(ML_SAMPLES.keys()):
        print(f"  - {filename}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            check_samples()
        elif sys.argv[1] == "--list":
            list_available()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python download_samples.py [--check|--list]")
    else:
        download_all()
