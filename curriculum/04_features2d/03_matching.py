"""
Module 4: Features2D - Feature Matching
=======================================
Matching features between images.

Official Docs: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

Topics Covered:
1. Brute-Force Matcher
2. FLANN Matcher
3. Ratio Test (Lowe's)
4. Homography Estimation
5. Object Detection with Features
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 4: Feature Matching")
print("=" * 60)


def create_test_images():
    """Create two related images for matching demo."""
    # Original image
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (200, 200), (255, 255, 255), 2)
    cv2.circle(img1, (300, 150), 50, (255, 255, 255), 2)
    cv2.putText(img1, "TEST", (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)

    # Add texture for better matching
    for i in range(10):
        x = np.random.randint(50, 350)
        y = np.random.randint(50, 250)
        cv2.circle(img1, (x, y), 5, (150, 150, 150), -1)

    # Transformed image (rotated and scaled)
    center = (200, 150)
    angle = 15
    scale = 0.9
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img2 = cv2.warpAffine(img1, M, (400, 300))

    # Add some noise
    noise = np.random.normal(0, 10, img2.shape).astype(np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img1, img2


img1, img2 = create_test_images()
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# =============================================================================
# 1. DETECTING FEATURES
# =============================================================================
print("\n--- 1. Detecting Features ---")

# Use ORB for this demo (free and fast)
orb = cv2.ORB_create(nfeatures=500)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

print(f"Image 1: {len(kp1)} keypoints")
print(f"Image 2: {len(kp2)} keypoints")


# =============================================================================
# 2. BRUTE-FORCE MATCHER
# =============================================================================
print("\n--- 2. Brute-Force Matcher ---")

# Create BF matcher
# For binary descriptors (ORB, BRISK): use NORM_HAMMING
# For float descriptors (SIFT): use NORM_L2

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

print(f"Total matches: {len(matches)}")
print(f"Best match distance: {matches[0].distance:.2f}")
print(f"Worst match distance: {matches[-1].distance:.2f}")

# Draw first 20 matches
bf_match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# =============================================================================
# 3. KNN MATCHING WITH RATIO TEST
# =============================================================================
print("\n--- 3. KNN Matching with Ratio Test ---")

# Create BF matcher without crossCheck for knnMatch
bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Find 2 best matches for each descriptor
matches_knn = bf_knn.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
# Keep match only if best match is significantly better than second best
ratio_thresh = 0.75
good_matches = []

for m, n in matches_knn:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

print(f"KNN matches: {len(matches_knn)}")
print(f"Good matches after ratio test: {len(good_matches)}")

# Draw good matches
knn_match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# =============================================================================
# 4. FLANN MATCHER
# =============================================================================
print("\n--- 4. FLANN Matcher ---")

# FLANN (Fast Library for Approximate Nearest Neighbors)
# Faster for large datasets

# For ORB (binary descriptors)
FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1
)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# For SIFT (float descriptors), use:
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)

try:
    flann_matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    flann_good = []
    for match_pair in flann_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                flann_good.append(m)

    print(f"FLANN good matches: {len(flann_good)}")
except cv2.error as e:
    print(f"FLANN error (may need more features): {e}")
    flann_good = good_matches  # Fallback


# =============================================================================
# 5. HOMOGRAPHY ESTIMATION
# =============================================================================
print("\n--- 5. Homography Estimation ---")

if len(good_matches) >= 4:
    # Extract matched keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matches_mask = mask.ravel().tolist()
    inliers = sum(matches_mask)

    print(f"Homography found")
    print(f"Inliers: {inliers} / {len(good_matches)}")

    # Draw bounding box of detected object
    h, w = gray1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    homography_img = img2.copy()
    cv2.polylines(homography_img, [np.int32(dst)], True, (0, 255, 0), 3)
else:
    print("Not enough matches for homography")
    homography_img = img2.copy()
    matches_mask = None


# =============================================================================
# 6. MATCH OBJECT PROPERTIES
# =============================================================================
print("\n--- 6. Match Properties ---")

if len(matches) > 0:
    m = matches[0]
    print(f"\nDMatch attributes:")
    print(f"  queryIdx: {m.queryIdx} (index in kp1)")
    print(f"  trainIdx: {m.trainIdx} (index in kp2)")
    print(f"  imgIdx: {m.imgIdx} (train image index)")
    print(f"  distance: {m.distance:.2f}")


# =============================================================================
# 7. MATCHING BEST PRACTICES
# =============================================================================
print("\n--- 7. Best Practices ---")

best_practices = """
Feature Matching Tips:

1. Choose the right detector/descriptor:
   - ORB: Fast, free, good for real-time
   - SIFT: More robust, slower

2. Use the ratio test:
   - Typically ratio = 0.75-0.8
   - Filters ambiguous matches

3. Use RANSAC for geometric verification:
   - Filters outliers
   - Required for homography/pose estimation

4. BF vs FLANN:
   - BF: Better for small datasets, guaranteed optimal
   - FLANN: Better for large datasets, approximate

5. Minimum matches needed:
   - Homography: 4 points minimum
   - Fundamental matrix: 8 points minimum
   - Always have more for robustness
"""
print(best_practices)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display matching demonstrations."""

    # Show BF matching
    cv2.putText(bf_match_img, "Brute-Force Matching (top 20)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("BF Matching", bf_match_img)

    # Show KNN matching with ratio test
    cv2.putText(knn_match_img, f"KNN + Ratio Test ({len(good_matches)} matches)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("KNN Matching", knn_match_img)

    # Show homography result
    cv2.putText(homography_img, "Detected Object (Homography)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Stack source and result
    homography_display = np.hstack([img1, homography_img])
    cv2.imshow("Homography Result", homography_display)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running feature matching demonstrations...")
    print("=" * 60)
    show_demo()
