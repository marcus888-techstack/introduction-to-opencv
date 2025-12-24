"""
Create Sample Images for OpenCV Course Projects
Run this script to generate all sample assets.
"""

import cv2
import numpy as np
from pathlib import Path
import urllib.request
import os

# Base paths
BASE_DIR = Path(__file__).parent
SAMPLE_IMAGES = BASE_DIR / "sample_images"
SAMPLE_VIDEOS = BASE_DIR / "sample_videos"

SAMPLE_IMAGES.mkdir(exist_ok=True)
SAMPLE_VIDEOS.mkdir(exist_ok=True)


def create_document_samples():
    """Create sample document images for Document Scanner project."""
    print("Creating document samples...")

    # Sample 1: Tilted document on dark background
    img = np.zeros((800, 1000, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark background

    # Draw tilted white document
    pts = np.array([[150, 100], [850, 80], [880, 700], [120, 720]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (250, 250, 250))

    # Add text content
    cv2.putText(img, "INVOICE", (250, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "Invoice No: INV-2024-001", (200, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.putText(img, "Date: December 24, 2024", (200, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.putText(img, "----------------------------", (200, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    cv2.putText(img, "Item 1: OpenCV Course     $99", (200, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    cv2.putText(img, "Item 2: Python Tutorial   $49", (200, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    cv2.putText(img, "----------------------------", (200, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    cv2.putText(img, "TOTAL:                   $148", (200, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    cv2.imwrite(str(SAMPLE_IMAGES / "document_tilted.jpg"), img)

    # Sample 2: Business card
    card = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(card, (10, 10), (490, 290), (200, 200, 200), 2)
    cv2.putText(card, "John Smith", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(card, "Software Engineer", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    cv2.putText(card, "Email: john@example.com", (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.putText(card, "Phone: +1 234 567 8900", (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.putText(card, "www.example.com", (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)

    # Place card on background with perspective
    bg = np.zeros((500, 700, 3), dtype=np.uint8)
    bg[:] = (60, 60, 80)
    bg[100:400, 100:600] = card

    cv2.imwrite(str(SAMPLE_IMAGES / "business_card.jpg"), bg)
    print(f"  Created: document_tilted.jpg, business_card.jpg")


def create_license_plate_samples():
    """Create sample license plate images."""
    print("Creating license plate samples...")

    def create_plate(text, style='us'):
        if style == 'us':
            plate = np.ones((120, 400, 3), dtype=np.uint8) * 255
            cv2.rectangle(plate, (5, 5), (395, 115), (0, 0, 150), 3)
            cv2.putText(plate, text, (40, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 100), 4)
        else:  # EU style
            plate = np.ones((100, 450, 3), dtype=np.uint8) * 255
            cv2.rectangle(plate, (0, 0), (40, 100), (0, 0, 200), -1)  # Blue strip
            cv2.rectangle(plate, (5, 5), (445, 95), (0, 0, 0), 2)
            cv2.putText(plate, text, (60, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
        return plate

    # Create plates
    plate1 = create_plate("ABC 1234", 'us')
    plate2 = create_plate("XY 12 ABC", 'eu')

    # Place on car-like backgrounds
    for i, (plate, name) in enumerate([(plate1, "plate_us.jpg"), (plate2, "plate_eu.jpg")]):
        # Create scene
        scene = np.zeros((500, 800, 3), dtype=np.uint8)
        scene[:] = (80, 80, 80)

        # Add car body (simplified rectangle)
        cv2.rectangle(scene, (100, 150), (700, 400), (50, 50, 60), -1)
        cv2.rectangle(scene, (150, 100), (650, 200), (40, 40, 50), -1)

        # Add headlights
        cv2.circle(scene, (180, 300), 30, (200, 200, 220), -1)
        cv2.circle(scene, (620, 300), 30, (200, 200, 220), -1)

        # Place plate
        ph, pw = plate.shape[:2]
        px, py = 300, 320
        scene[py:py+ph, px:px+pw] = plate

        cv2.imwrite(str(SAMPLE_IMAGES / name), scene)

    print(f"  Created: plate_us.jpg, plate_eu.jpg")


def create_quality_inspection_samples():
    """Create sample images for Quality Inspection project."""
    print("Creating quality inspection samples...")

    # Reference (good) sample
    good = np.ones((400, 400, 3), dtype=np.uint8) * 200
    cv2.rectangle(good, (50, 50), (350, 350), (180, 180, 180), -1)
    # Add some texture
    for i in range(50, 350, 20):
        cv2.line(good, (i, 50), (i, 350), (175, 175, 175), 1)
    cv2.imwrite(str(SAMPLE_IMAGES / "quality_good.jpg"), good)

    # Defective sample 1: Spots
    defect1 = good.copy()
    cv2.circle(defect1, (150, 150), 15, (50, 50, 50), -1)
    cv2.circle(defect1, (280, 200), 12, (60, 60, 60), -1)
    cv2.circle(defect1, (200, 300), 8, (70, 70, 70), -1)
    cv2.imwrite(str(SAMPLE_IMAGES / "quality_spots.jpg"), defect1)

    # Defective sample 2: Scratch/Crack
    defect2 = good.copy()
    pts = np.array([[80, 100], [150, 180], [250, 150], [320, 200], [350, 180]])
    cv2.polylines(defect2, [pts], False, (30, 30, 30), 3)
    cv2.imwrite(str(SAMPLE_IMAGES / "quality_crack.jpg"), defect2)

    # Defective sample 3: Multiple issues
    defect3 = good.copy()
    cv2.circle(defect3, (100, 100), 20, (40, 40, 40), -1)
    pts = np.array([[200, 250], [250, 280], [300, 260], [350, 300]])
    cv2.polylines(defect3, [pts], False, (35, 35, 35), 2)
    cv2.rectangle(defect3, (280, 80), (320, 120), (45, 45, 45), -1)
    cv2.imwrite(str(SAMPLE_IMAGES / "quality_multiple.jpg"), defect3)

    print(f"  Created: quality_good.jpg, quality_spots.jpg, quality_crack.jpg, quality_multiple.jpg")


def create_face_samples():
    """Create placeholder face images (synthetic/cartoon style)."""
    print("Creating face sample placeholders...")

    def draw_face(name, color):
        img = np.ones((300, 300, 3), dtype=np.uint8) * 240

        # Face oval
        cv2.ellipse(img, (150, 150), (80, 100), 0, 0, 360, color, -1)

        # Eyes
        cv2.circle(img, (120, 130), 12, (255, 255, 255), -1)
        cv2.circle(img, (180, 130), 12, (255, 255, 255), -1)
        cv2.circle(img, (120, 130), 5, (0, 0, 0), -1)
        cv2.circle(img, (180, 130), 5, (0, 0, 0), -1)

        # Nose
        pts = np.array([[150, 140], [140, 170], [160, 170]])
        cv2.polylines(img, [pts], False, (100, 100, 100), 2)

        # Mouth
        cv2.ellipse(img, (150, 200), (30, 15), 0, 0, 180, (100, 100, 100), 2)

        # Name
        cv2.putText(img, name, (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return img

    # Create sample faces
    faces = [
        ("Alice", (180, 200, 230)),
        ("Bob", (200, 180, 160)),
        ("Charlie", (190, 210, 200)),
    ]

    for name, color in faces:
        face_img = draw_face(name, color)
        cv2.imwrite(str(SAMPLE_IMAGES / f"face_{name.lower()}.jpg"), face_img)

    print(f"  Created: face_alice.jpg, face_bob.jpg, face_charlie.jpg")


def create_counting_video():
    """Create a simple video with moving objects for counting demo."""
    print("Creating object counting video...")

    width, height = 800, 600
    fps = 30
    duration = 5  # seconds

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(SAMPLE_VIDEOS / "counting_demo.mp4"),
                          fourcc, fps, (width, height))

    # Simulated objects
    objects = [
        {'x': 100, 'y': 100, 'vx': 5, 'vy': 3, 'size': 40, 'color': (0, 0, 255)},
        {'x': 300, 'y': 50, 'vx': 3, 'vy': 4, 'size': 35, 'color': (0, 255, 0)},
        {'x': 500, 'y': 150, 'vx': -4, 'vy': 5, 'size': 45, 'color': (255, 0, 0)},
        {'x': 200, 'y': 400, 'vx': 6, 'vy': -2, 'size': 38, 'color': (255, 255, 0)},
        {'x': 600, 'y': 300, 'vx': -3, 'vy': 4, 'size': 42, 'color': (255, 0, 255)},
    ]

    for frame_num in range(fps * duration):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50

        # Draw counting line
        cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 255), 2)
        cv2.putText(frame, "Counting Line", (10, height // 2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Update and draw objects
        for obj in objects:
            # Update position
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']

            # Bounce off walls
            if obj['x'] <= 0 or obj['x'] >= width - obj['size']:
                obj['vx'] *= -1
            if obj['y'] <= 0 or obj['y'] >= height - obj['size']:
                obj['vy'] *= -1

            # Keep in bounds
            obj['x'] = max(0, min(width - obj['size'], obj['x']))
            obj['y'] = max(0, min(height - obj['size'], obj['y']))

            # Draw object
            cv2.rectangle(frame,
                         (int(obj['x']), int(obj['y'])),
                         (int(obj['x'] + obj['size']), int(obj['y'] + obj['size'])),
                         obj['color'], -1)

        out.write(frame)

    out.release()
    print(f"  Created: counting_demo.mp4 ({duration}s, {fps}fps)")


def download_sample_video():
    """Download a sample pedestrian video from Pexels (optional)."""
    print("Attempting to download sample video from Pexels...")

    # This is a direct link to a small sample video
    # Note: For production, use Pexels API with proper attribution
    sample_urls = [
        ("https://videos.pexels.com/video-files/3153846/3153846-sd_640_360_30fps.mp4",
         "pedestrian_sample.mp4"),
    ]

    for url, filename in sample_urls:
        output_path = SAMPLE_VIDEOS / filename
        if output_path.exists():
            print(f"  {filename} already exists, skipping...")
            continue

        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, str(output_path))
            print(f"  Downloaded: {filename}")
        except Exception as e:
            print(f"  Could not download {filename}: {e}")
            print(f"  Using generated demo video instead.")


def create_all_samples():
    """Create all sample assets."""
    print("\n" + "=" * 50)
    print("Creating Sample Assets for OpenCV Course")
    print("=" * 50 + "\n")

    create_document_samples()
    create_license_plate_samples()
    create_quality_inspection_samples()
    create_face_samples()
    create_counting_video()

    # Try to download real video (optional)
    try:
        download_sample_video()
    except Exception as e:
        print(f"  Video download skipped: {e}")

    print("\n" + "=" * 50)
    print("All samples created successfully!")
    print("=" * 50)
    print(f"\nImages: {SAMPLE_IMAGES}")
    print(f"Videos: {SAMPLE_VIDEOS}")

    # List created files
    print("\nCreated files:")
    for f in sorted(SAMPLE_IMAGES.glob("*")):
        print(f"  {f.name}")
    for f in sorted(SAMPLE_VIDEOS.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    create_all_samples()
