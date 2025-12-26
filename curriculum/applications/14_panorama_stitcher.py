"""
Application 14: Panorama Stitcher
================================
Create panoramic images by stitching multiple photos together.

Techniques Used:
- Feature detection (ORB/SIFT)
- Feature matching
- Homography estimation
- Image warping and blending

Official Docs:
- https://docs.opencv.org/4.x/d1/d46/group__stitching.html
- https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image


class PanoramaStitcher:
    """
    Create panoramas from multiple images.
    """

    def __init__(self, feature_type='orb'):
        # Feature detector
        if feature_type == 'sift':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:  # ORB (faster, free)
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.feature_type = feature_type

        # Stitcher mode
        self.blend_mode = 'average'  # 'average', 'feather', 'multiband'

    def detect_and_match(self, img1, img2):
        """
        Detect features and find matches between two images.
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return None, None, None

        # Match features
        if self.feature_type == 'sift':
            matches = self.matcher.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        else:
            matches = self.matcher.match(des1, des2)
            # Sort by distance
            matches = sorted(matches, key=lambda x: x.distance)
            # Keep best matches
            good_matches = matches[:int(len(matches) * 0.7)]

        return kp1, kp2, good_matches

    def find_homography(self, kp1, kp2, matches, min_matches=4):
        """
        Find homography matrix from matched keypoints.
        """
        if len(matches) < min_matches:
            return None, None

        # Extract matching point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return H, mask

    def warp_and_blend(self, img1, img2, H):
        """
        Warp img1 to img2's perspective and blend them.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Get corners of img1 after transformation
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners1_warped = cv2.perspectiveTransform(corners1, H)

        # Combine with img2 corners
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        all_corners = np.concatenate([corners1_warped, corners2], axis=0)

        # Find bounding box
        x_min = int(np.floor(all_corners[:, 0, 0].min()))
        x_max = int(np.ceil(all_corners[:, 0, 0].max()))
        y_min = int(np.floor(all_corners[:, 0, 1].min()))
        y_max = int(np.ceil(all_corners[:, 0, 1].max()))

        # Translation matrix
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        # Output dimensions
        output_w = x_max - x_min
        output_h = y_max - y_min

        # Warp img1
        H_translated = translation @ H
        warped1 = cv2.warpPerspective(img1, H_translated, (output_w, output_h))

        # Place img2
        warped2 = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        warped2[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2

        # Blend
        result = self._blend_images(warped1, warped2)

        return result

    def _blend_images(self, img1, img2):
        """
        Blend two warped images.
        """
        # Create masks
        mask1 = (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
        mask2 = (cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

        # Overlap region
        overlap = mask1 * mask2

        if self.blend_mode == 'average':
            # Simple average in overlap region
            result = img1.copy().astype(np.float32)

            for c in range(3):
                # Non-overlap: use whichever image has content
                result[:, :, c] = np.where(overlap > 0,
                    (img1[:, :, c].astype(np.float32) + img2[:, :, c].astype(np.float32)) / 2,
                    np.where(mask1 > 0, img1[:, :, c], img2[:, :, c]))

        elif self.blend_mode == 'feather':
            # Distance-based feathering
            dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)

            # Normalize weights
            total = dist1 + dist2 + 1e-6
            weight1 = dist1 / total
            weight2 = dist2 / total

            result = np.zeros_like(img1, dtype=np.float32)
            for c in range(3):
                result[:, :, c] = img1[:, :, c] * weight1 + img2[:, :, c] * weight2

        else:
            # Default: prefer img2 (reference)
            result = np.where(mask2[:, :, np.newaxis] > 0, img2, img1).astype(np.float32)

        return result.astype(np.uint8)

    def stitch(self, images):
        """
        Stitch multiple images into a panorama.
        """
        if len(images) < 2:
            return images[0] if images else None

        # Start with first image
        result = images[0]

        for i in range(1, len(images)):
            print(f"Stitching image {i+1}/{len(images)}...")

            # Detect and match features
            kp1, kp2, matches = self.detect_and_match(images[i], result)

            if matches is None or len(matches) < 4:
                print(f"Warning: Not enough matches for image {i+1}")
                continue

            print(f"  Found {len(matches)} matches")

            # Find homography
            H, mask = self.find_homography(kp1, kp2, matches)

            if H is None:
                print(f"Warning: Could not find homography for image {i+1}")
                continue

            # Warp and blend
            result = self.warp_and_blend(images[i], result, H)

        return result

    def draw_matches(self, img1, img2, kp1, kp2, matches, max_show=50):
        """
        Draw matches between two images.
        """
        # Limit matches for visualization
        matches_to_show = matches[:max_show]

        result = cv2.drawMatches(img1, kp1, img2, kp2, matches_to_show, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return result


def use_opencv_stitcher(images):
    """
    Use OpenCV's built-in Stitcher class.
    """
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return pano
    else:
        status_names = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera params adjust failed"
        }
        print(f"Stitching failed: {status_names.get(status, 'Unknown error')}")
        return None


def load_panorama_images():
    """
    Load images for panorama or create synthetic set.
    """
    images = []

    # Try to load panorama set
    for i in range(1, 5):
        for pattern in [f"panorama_{i}.jpg", f"pano{i}.jpg", f"scene_{i}.jpg"]:
            img = get_image(pattern)
            if img is not None:
                images.append(img)
                print(f"Loaded: {pattern}")
                break

    if len(images) >= 2:
        return images

    # Create synthetic panorama set
    print("No panorama images found. Creating synthetic set.")

    # Create a large scene
    scene = np.zeros((400, 1200, 3), dtype=np.uint8)

    # Background gradient
    for x in range(1200):
        scene[:, x] = (
            int(100 + (x / 1200) * 100),
            int(150 - (x / 1200) * 50),
            int(200 - (x / 1200) * 100)
        )

    # Add some shapes
    cv2.rectangle(scene, (100, 150), (250, 350), (0, 100, 200), -1)
    cv2.circle(scene, (400, 200), 80, (200, 100, 0), -1)
    cv2.rectangle(scene, (550, 100), (700, 300), (0, 200, 100), -1)
    cv2.circle(scene, (900, 250), 100, (100, 0, 200), -1)
    cv2.rectangle(scene, (1000, 150), (1150, 350), (200, 200, 0), -1)

    # Add text
    cv2.putText(scene, "PANORAMA SCENE", (450, 380),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Extract overlapping sections
    width = 500
    overlap = 150

    images = [
        scene[:, 0:width],
        scene[:, width-overlap:2*width-overlap],
        scene[:, 2*width-2*overlap:]
    ]

    # Add slight variations
    for i, img in enumerate(images):
        # Add some noise
        noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
        images[i] = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return images


def interactive_stitcher():
    """
    Interactive panorama creation.
    """
    print("\n=== Panorama Stitcher ===")
    print("Controls:")
    print("  'c' - Capture frame for panorama (webcam)")
    print("  's' - Stitch captured images")
    print("  'o' - Use OpenCV Stitcher")
    print("  'r' - Reset/clear images")
    print("  'f' - Toggle feature type (ORB/SIFT)")
    print("  'q' - Quit")
    print("=========================\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Using demo mode.")
        demo_mode()
        return

    stitcher = PanoramaStitcher('orb')
    captured_images = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display frame
        display = frame.copy()
        cv2.putText(display, f"Captured: {len(captured_images)} | Feature: {stitcher.feature_type}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "Press 'c' to capture, 's' to stitch",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Panorama Capture", display)

        # Show captured images thumbnails
        if captured_images:
            thumbs = []
            for img in captured_images:
                thumb = cv2.resize(img, (160, 120))
                thumbs.append(thumb)
            if len(thumbs) <= 5:
                thumb_strip = np.hstack(thumbs)
                cv2.imshow("Captured Images", thumb_strip)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            captured_images.append(frame.copy())
            print(f"Captured image {len(captured_images)}")
        elif key == ord('s') and len(captured_images) >= 2:
            print("Stitching with custom stitcher...")
            result = stitcher.stitch(captured_images)
            if result is not None:
                cv2.imshow("Panorama Result", result)
                cv2.imwrite("panorama_result.jpg", result)
                print("Saved: panorama_result.jpg")
        elif key == ord('o') and len(captured_images) >= 2:
            print("Stitching with OpenCV Stitcher...")
            result = use_opencv_stitcher(captured_images)
            if result is not None:
                cv2.imshow("Panorama (OpenCV)", result)
                cv2.imwrite("panorama_opencv.jpg", result)
                print("Saved: panorama_opencv.jpg")
        elif key == ord('r'):
            captured_images = []
            cv2.destroyWindow("Captured Images")
            print("Cleared captured images")
        elif key == ord('f'):
            new_type = 'sift' if stitcher.feature_type == 'orb' else 'orb'
            stitcher = PanoramaStitcher(new_type)
            print(f"Switched to {new_type.upper()}")

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    """
    Demo with pre-loaded or synthetic images.
    """
    print("\n=== Panorama Stitcher Demo ===\n")

    # Load images
    images = load_panorama_images()
    print(f"Loaded {len(images)} images for stitching")

    if len(images) < 2:
        print("Need at least 2 images")
        return

    stitcher = PanoramaStitcher('orb')

    # Show input images
    thumbs = [cv2.resize(img, (200, 150)) for img in images]
    input_display = np.hstack(thumbs)
    cv2.putText(input_display, "Input Images", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Input Images", input_display)

    # Show feature matches for first pair
    kp1, kp2, matches = stitcher.detect_and_match(images[0], images[1])
    if matches:
        match_img = stitcher.draw_matches(images[0], images[1], kp1, kp2, matches)
        match_img = cv2.resize(match_img, (800, 300))
        cv2.imshow("Feature Matches", match_img)
        print(f"Found {len(matches)} matches between first two images")

    # Stitch panorama
    print("\nStitching panorama...")
    result = stitcher.stitch(images)

    if result is not None:
        cv2.imshow("Panorama Result", result)
        cv2.imwrite("panorama_result.jpg", result)
        print("Saved: panorama_result.jpg")

    # Also try OpenCV stitcher
    print("\nTrying OpenCV Stitcher...")
    opencv_result = use_opencv_stitcher(images)
    if opencv_result is not None:
        cv2.imshow("Panorama (OpenCV)", opencv_result)

    print("\nPanorama pipeline:")
    print("1. Detect features in each image")
    print("2. Match features between overlapping images")
    print("3. Estimate homography transforms")
    print("4. Warp images to common plane")
    print("5. Blend seamlessly")

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Application 14: Panorama Stitcher")
    print("=" * 60)

    try:
        interactive_stitcher()
    except Exception as e:
        print(f"Error: {e}")
        demo_mode()
