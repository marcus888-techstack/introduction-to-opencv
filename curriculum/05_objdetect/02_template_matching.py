"""
Module 5: Object Detection - Template Matching
===============================================
Finding objects using template matching.

Official Docs: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

Topics Covered:
1. Basic Template Matching
2. Matching Methods
3. Multiple Object Detection
4. Scale-Invariant Matching
5. Rotation Handling
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 5: Template Matching")
print("=" * 60)


def create_test_images():
    """Create test image and template."""
    # Main image with multiple shapes
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    # Add stars (our target object)
    def draw_star(img, center, size, color):
        pts = []
        for i in range(5):
            angle = i * 72 - 90  # Start from top
            rad = np.radians(angle)
            x = int(center[0] + size * np.cos(rad))
            y = int(center[1] + size * np.sin(rad))
            pts.append([x, y])
            # Inner point
            inner_angle = angle + 36
            inner_rad = np.radians(inner_angle)
            inner_x = int(center[0] + size * 0.4 * np.cos(inner_rad))
            inner_y = int(center[1] + size * 0.4 * np.sin(inner_rad))
            pts.append([inner_x, inner_y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color)

    # Draw multiple stars
    star_positions = [(100, 100), (300, 150), (500, 100), (200, 300), (450, 280)]
    for pos in star_positions:
        draw_star(img, pos, 40, (255, 255, 0))

    # Add some other shapes (distractors)
    cv2.rectangle(img, (350, 220), (400, 270), (0, 255, 0), -1)
    cv2.circle(img, (150, 280), 30, (255, 0, 0), -1)

    # Create template (single star)
    template = np.zeros((100, 100, 3), dtype=np.uint8)
    template[:] = (50, 50, 50)
    draw_star(template, (50, 50), 40, (255, 255, 0))

    return img, template


img, template = create_test_images()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

h, w = gray_template.shape


# =============================================================================
# 1. BASIC TEMPLATE MATCHING
# =============================================================================
print("\n--- 1. Basic Template Matching ---")

# Match template
result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)

# Find best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(f"Result shape: {result.shape}")
print(f"Best match value: {max_val:.4f}")
print(f"Best match location: {max_loc}")

# For TM_CCOEFF_NORMED, max_loc is best match
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw rectangle
match_img = img.copy()
cv2.rectangle(match_img, top_left, bottom_right, (0, 255, 0), 2)


# =============================================================================
# 2. MATCHING METHODS
# =============================================================================
print("\n--- 2. Matching Methods ---")

methods = [
    ('TM_SQDIFF', cv2.TM_SQDIFF),
    ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED),
    ('TM_CCORR', cv2.TM_CCORR),
    ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
    ('TM_CCOEFF', cv2.TM_CCOEFF),
    ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
]

print("\nMethod comparison:")
for name, method in methods:
    result = cv2.matchTemplate(gray_img, gray_template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # For SQDIFF methods, minimum is best match
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        match_loc = min_loc
        match_val = min_val
    else:
        match_loc = max_loc
        match_val = max_val

    print(f"  {name}: value={match_val:.4f}, loc={match_loc}")

methods_info = """
Matching Methods:
  TM_SQDIFF         - Sum of squared differences (min = best)
  TM_SQDIFF_NORMED  - Normalized squared diff (min = best)
  TM_CCORR          - Cross correlation (max = best)
  TM_CCORR_NORMED   - Normalized cross correlation (max = best)
  TM_CCOEFF         - Correlation coefficient (max = best)
  TM_CCOEFF_NORMED  - Normalized corr coeff (max = best)

Recommendations:
  - TM_CCOEFF_NORMED: Best for most cases (value 0-1)
  - TM_SQDIFF_NORMED: Good alternative (value 0-1, lower is better)
"""
print(methods_info)


# =============================================================================
# 3. MULTIPLE OBJECT DETECTION
# =============================================================================
print("\n--- 3. Multiple Object Detection ---")

# Use normalized method for thresholding
result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)

# Find all matches above threshold
threshold = 0.8
locations = np.where(result >= threshold)

print(f"Threshold: {threshold}")
print(f"Matches found: {len(locations[0])}")

# Draw all matches
multi_match_img = img.copy()
for pt in zip(*locations[::-1]):  # Switch x and y
    cv2.rectangle(multi_match_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)


# =============================================================================
# 4. NON-MAXIMUM SUPPRESSION
# =============================================================================
print("\n--- 4. Non-Maximum Suppression ---")


def non_max_suppression(boxes, scores, threshold=0.5):
    """Apply non-maximum suppression to remove overlapping boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)

    # Sort by scores
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate IoU with rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_overlap = np.maximum(0, xx2 - xx1)
        h_overlap = np.maximum(0, yy2 - yy1)

        overlap = w_overlap * h_overlap
        iou = overlap / (areas[i] + areas[order[1:]] - overlap)

        # Keep boxes with low IoU
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


# Get boxes and scores
boxes = []
scores = []
for pt in zip(*locations[::-1]):
    boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
    scores.append(result[pt[1], pt[0]])

# Apply NMS
if len(boxes) > 0:
    keep_indices = non_max_suppression(boxes, scores, threshold=0.3)
    print(f"After NMS: {len(keep_indices)} objects")

    nms_img = img.copy()
    for i in keep_indices:
        box = boxes[i]
        cv2.rectangle(nms_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
else:
    nms_img = img.copy()


# =============================================================================
# 5. SCALE-INVARIANT MATCHING
# =============================================================================
print("\n--- 5. Multi-Scale Matching ---")


def multi_scale_template_matching(image, template, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """Match template at multiple scales."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

    best_match = None
    best_val = -1
    best_scale = 1.0

    for scale in scales:
        # Resize template
        resized = cv2.resize(templ_gray, None, fx=scale, fy=scale)

        # Skip if template larger than image
        if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
            continue

        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_match = (max_loc, resized.shape[1], resized.shape[0])
            best_scale = scale

    return best_match, best_val, best_scale


match, val, scale = multi_scale_template_matching(img, template)
print(f"Best scale: {scale}, value: {val:.4f}")

if match:
    scale_img = img.copy()
    x, y = match[0]
    w, h = match[1], match[2]
    cv2.rectangle(scale_img, (x, y), (x + w, y + h), (255, 0, 255), 2)


# =============================================================================
# 6. LIMITATIONS
# =============================================================================
print("\n--- 6. Template Matching Limitations ---")

limitations = """
Template Matching Limitations:

1. Rotation Sensitivity:
   - Template must match orientation
   - Solution: Try multiple rotated templates

2. Scale Sensitivity:
   - Fixed template size
   - Solution: Multi-scale matching

3. Lighting Changes:
   - Affected by brightness differences
   - Solution: Normalize images, use TM_CCOEFF_NORMED

4. Occlusion:
   - Cannot handle partial visibility
   - Solution: Use feature-based detection

5. Deformation:
   - Cannot handle non-rigid objects
   - Solution: Use more robust methods (DNN, features)

Best Use Cases:
- Fixed-size objects
- Consistent lighting
- No rotation (or known rotation)
- Quick prototyping
"""
print(limitations)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display template matching demos."""

    # Template
    cv2.imshow("Template", template)

    # Single best match
    cv2.putText(match_img, f"Best: {max_val:.2f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Best Match", match_img)

    # Multiple matches
    cv2.putText(multi_match_img, f"All matches (threshold={threshold})", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Multiple Matches", multi_match_img)

    # After NMS
    cv2.putText(nms_img, "After NMS", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("NMS Result", nms_img)

    # Result heatmap
    result_display = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result_display = result_display.astype(np.uint8)
    result_color = cv2.applyColorMap(result_display, cv2.COLORMAP_JET)
    cv2.imshow("Match Heatmap", result_color)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running template matching demonstrations...")
    print("=" * 60)
    show_demo()
