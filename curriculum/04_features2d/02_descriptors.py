"""
Module 4: Features2D - Feature Descriptors
==========================================
ORB, SIFT, and other feature descriptors.

Official Docs: https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html

Topics Covered:
1. ORB (Oriented FAST and Rotated BRIEF)
2. SIFT (Scale-Invariant Feature Transform)
3. BRISK
4. AKAZE
5. Keypoint Visualization
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 4: Feature Descriptors")
print("=" * 60)


def create_test_image():
    """Create a test image with features."""
    img = np.zeros((400, 500, 3), dtype=np.uint8)

    # Background texture
    for i in range(0, 500, 20):
        cv2.line(img, (i, 0), (i, 400), (30, 30, 30), 1)
    for i in range(0, 400, 20):
        cv2.line(img, (0, i), (500, i), (30, 30, 30), 1)

    # Various shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), 2)
    cv2.circle(img, (250, 100), 50, (255, 255, 255), 2)
    cv2.ellipse(img, (400, 100), (40, 60), 30, 0, 360, (255, 255, 255), 2)

    # Text
    cv2.putText(img, "OpenCV", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)

    # Complex shape
    pts = np.array([[100, 300], [150, 350], [100, 380], [50, 350]], np.int32)
    cv2.polylines(img, [pts], True, (255, 255, 255), 2)

    # Star pattern
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        x = int(400 + 60 * np.cos(rad))
        y = int(300 + 60 * np.sin(rad))
        cv2.line(img, (400, 300), (x, y), (200, 200, 200), 2)

    return img


original = create_test_image()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)


# =============================================================================
# 1. ORB (Oriented FAST and Rotated BRIEF)
# =============================================================================
print("\n--- 1. ORB Descriptor ---")

# Create ORB detector
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and compute descriptors
keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)

print(f"ORB keypoints: {len(keypoints_orb)}")
if descriptors_orb is not None:
    print(f"ORB descriptor shape: {descriptors_orb.shape}")
    print(f"ORB descriptor type: {descriptors_orb.dtype} (binary)")

# ORB parameters
orb_custom = cv2.ORB_create(
    nfeatures=1000,           # Max features
    scaleFactor=1.2,          # Pyramid scale
    nlevels=8,                # Pyramid levels
    edgeThreshold=31,         # Border margin
    firstLevel=0,             # First pyramid level
    WTA_K=2,                  # Points for BRIEF (2,3,4)
    scoreType=cv2.ORB_HARRIS_SCORE,  # Scoring method
    patchSize=31,             # BRIEF patch size
    fastThreshold=20          # FAST threshold
)

print("ORB: Free, fast, good for real-time applications")


# =============================================================================
# 2. SIFT (Scale-Invariant Feature Transform)
# =============================================================================
print("\n--- 2. SIFT Descriptor ---")

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect and compute
keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)

print(f"SIFT keypoints: {len(keypoints_sift)}")
if descriptors_sift is not None:
    print(f"SIFT descriptor shape: {descriptors_sift.shape}")
    print(f"SIFT descriptor type: {descriptors_sift.dtype} (float)")

# SIFT parameters
sift_custom = cv2.SIFT_create(
    nfeatures=0,              # Max features (0=no limit)
    nOctaveLayers=3,          # Layers per octave
    contrastThreshold=0.04,   # Low contrast filter
    edgeThreshold=10,         # Edge filter
    sigma=1.6                 # Gaussian sigma
)

print("SIFT: Very robust, scale/rotation invariant, but slower")


# =============================================================================
# 3. BRISK (Binary Robust Invariant Scalable Keypoints)
# =============================================================================
print("\n--- 3. BRISK Descriptor ---")

# Create BRISK detector
brisk = cv2.BRISK_create()

# Detect and compute
keypoints_brisk, descriptors_brisk = brisk.detectAndCompute(gray, None)

print(f"BRISK keypoints: {len(keypoints_brisk)}")
if descriptors_brisk is not None:
    print(f"BRISK descriptor shape: {descriptors_brisk.shape}")

# BRISK parameters
brisk_custom = cv2.BRISK_create(
    thresh=30,                # AGAST detection threshold
    octaves=3,                # Detection octaves
    patternScale=1.0          # Pattern scale
)

print("BRISK: Binary descriptor, fast, scale-invariant")


# =============================================================================
# 4. AKAZE (Accelerated-KAZE)
# =============================================================================
print("\n--- 4. AKAZE Descriptor ---")

# Create AKAZE detector
akaze = cv2.AKAZE_create()

# Detect and compute
keypoints_akaze, descriptors_akaze = akaze.detectAndCompute(gray, None)

print(f"AKAZE keypoints: {len(keypoints_akaze)}")
if descriptors_akaze is not None:
    print(f"AKAZE descriptor shape: {descriptors_akaze.shape}")

# AKAZE parameters
akaze_custom = cv2.AKAZE_create(
    descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,  # Descriptor type
    descriptor_size=0,        # 0 = full size
    descriptor_channels=3,    # Descriptor channels
    threshold=0.001,          # Detector threshold
    nOctaves=4,               # Octaves
    nOctaveLayers=4           # Layers per octave
)

print("AKAZE: Good for deformable objects, built on nonlinear scale space")


# =============================================================================
# 5. KEYPOINT PROPERTIES
# =============================================================================
print("\n--- 5. Keypoint Properties ---")

if len(keypoints_orb) > 0:
    kp = keypoints_orb[0]
    print(f"\nKeypoint attributes:")
    print(f"  pt (position): ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})")
    print(f"  size: {kp.size:.2f}")
    print(f"  angle: {kp.angle:.2f}")
    print(f"  response: {kp.response:.4f}")
    print(f"  octave: {kp.octave}")
    print(f"  class_id: {kp.class_id}")


# =============================================================================
# 6. ALGORITHM COMPARISON
# =============================================================================
print("\n--- 6. Algorithm Comparison ---")

comparison = """
Feature Descriptor Comparison:

| Algorithm | Speed   | Desc Size | Type   | Best For                |
|-----------|---------|-----------|--------|-------------------------|
| ORB       | Fast    | 32 bytes  | Binary | Real-time, mobile       |
| SIFT      | Slow    | 128 floats| Float  | High accuracy needs     |
| BRISK     | Fast    | 64 bytes  | Binary | Scale-invariant matching|
| AKAZE     | Medium  | Variable  | Binary | Deformable objects      |

Binary descriptors (ORB, BRISK, AKAZE):
  - Use Hamming distance for matching
  - Faster to compare
  - Smaller memory footprint

Float descriptors (SIFT):
  - Use Euclidean distance for matching
  - More accurate but slower
  - Larger memory footprint
"""
print(comparison)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display feature detection demos."""

    # Draw keypoints for each detector
    orb_img = cv2.drawKeypoints(original, keypoints_orb, None,
                                 color=(0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(orb_img, f"ORB: {len(keypoints_orb)} pts", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    sift_img = cv2.drawKeypoints(original, keypoints_sift, None,
                                  color=(255, 0, 0),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(sift_img, f"SIFT: {len(keypoints_sift)} pts", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    brisk_img = cv2.drawKeypoints(original, keypoints_brisk, None,
                                   color=(0, 0, 255),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(brisk_img, f"BRISK: {len(keypoints_brisk)} pts", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    akaze_img = cv2.drawKeypoints(original, keypoints_akaze, None,
                                   color=(255, 255, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(akaze_img, f"AKAZE: {len(keypoints_akaze)} pts", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Stack displays
    row1 = np.hstack([orb_img, sift_img])
    row2 = np.hstack([brisk_img, akaze_img])
    display = np.vstack([row1, row2])
    display = cv2.resize(display, (1000, 800))

    cv2.imshow("Feature Descriptors Comparison", display)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running feature descriptor demonstrations...")
    print("=" * 60)
    show_demo()
