"""
Module 1: Core Functionality (cv2.core)
=======================================
The foundation of OpenCV - understanding Mat, basic operations, and pixel manipulation.

Official Docs: https://docs.opencv.org/4.x/d0/de1/group__core.html

Topics Covered:
1. Mat structure (numpy arrays in Python)
2. Creating images
3. Pixel access and manipulation
4. Basic arithmetic operations
5. Logical operations
6. Splitting and merging channels
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 1: OpenCV Core Functionality")
print("=" * 60)


# =============================================================================
# 1. CREATING IMAGES (Mat objects)
# =============================================================================
print("\n--- 1. Creating Images ---")

# Create a black image (zeros)
black_img = np.zeros((300, 400, 3), dtype=np.uint8)
print(f"Black image shape: {black_img.shape}")  # (height, width, channels)
print(f"Data type: {black_img.dtype}")

# Create a white image (ones * 255)
white_img = np.ones((300, 400, 3), dtype=np.uint8) * 255

# Create a colored image
blue_img = np.zeros((300, 400, 3), dtype=np.uint8)
blue_img[:] = (255, 0, 0)  # BGR format!

red_img = np.zeros((300, 400, 3), dtype=np.uint8)
red_img[:] = (0, 0, 255)

green_img = np.zeros((300, 400, 3), dtype=np.uint8)
green_img[:] = (0, 255, 0)

# Create grayscale image
gray_img = np.zeros((300, 400), dtype=np.uint8)  # 2D, no channels
gray_img[:] = 128  # Mid-gray (128, 128, 128)


# =============================================================================
# 2. PIXEL ACCESS AND MANIPULATION
# =============================================================================
print("\n--- 2. Pixel Access ---")

# Create test image
img = np.zeros((100, 100, 3), dtype=np.uint8)

# Access single pixel (row, col) - returns BGR values
pixel = img[50, 50]
print(f"Pixel at (50,50): {pixel}")

# Modify single pixel
img[50, 50] = (0, 255, 0)  # Set to green
print(f"Modified pixel: {img[50, 50]}")

# Access only blue channel
blue_value = img[50, 50, 0]
print(f"Blue channel value: {blue_value}")

# Modify region (draw a red rectangle manually)
img[20:40, 20:80] = (0, 0, 255)  # Red rectangle

# Using itemget/itemset (faster for single pixels)
b = img.item(50, 50, 0)  # Blue channel
g = img.item(50, 50, 1)  # Green channel
r = img.item(50, 50, 2)  # Red channel
print(f"BGR values using item(): B={b}, G={g}, R={r}")


# =============================================================================
# 3. IMAGE PROPERTIES
# =============================================================================
print("\n--- 3. Image Properties ---")

sample = np.zeros((480, 640, 3), dtype=np.uint8)

print(f"Shape: {sample.shape}")           # (rows, cols, channels)
print(f"Height: {sample.shape[0]}")       # 480
print(f"Width: {sample.shape[1]}")        # 640
print(f"Channels: {sample.shape[2]}")     # 3
print(f"Total pixels: {sample.size}")     # 480*640*3
print(f"Data type: {sample.dtype}")       # uint8


# =============================================================================
# 4. ARITHMETIC OPERATIONS
# =============================================================================
print("\n--- 4. Arithmetic Operations ---")

# Create two images
img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
img2 = np.ones((100, 100, 3), dtype=np.uint8) * 200

# cv2.add - saturates at 255 (doesn't overflow)
result_add = cv2.add(img1, img2)
print(f"cv2.add(100, 200) = {result_add[0, 0, 0]}")  # 255 (saturated)

# NumPy add - wraps around (overflows)
result_np = img1 + img2
print(f"numpy add(100, 200) = {result_np[0, 0, 0]}")  # 44 (wrapped: 300-256)

# Subtraction
result_sub = cv2.subtract(img2, img1)
print(f"cv2.subtract(200, 100) = {result_sub[0, 0, 0]}")  # 100

# Weighted addition (blending)
# dst = α*img1 + β*img2 + γ
alpha = 0.7
beta = 0.3
gamma = 0
blended = cv2.addWeighted(img1, alpha, img2, beta, gamma)
print(f"Weighted blend: 0.7*100 + 0.3*200 = {blended[0, 0, 0]}")  # 130


# =============================================================================
# 5. LOGICAL (BITWISE) OPERATIONS
# =============================================================================
print("\n--- 5. Bitwise Operations ---")

# Create two binary-like images
rect1 = np.zeros((200, 200), dtype=np.uint8)
rect2 = np.zeros((200, 200), dtype=np.uint8)

cv2.rectangle(rect1, (20, 20), (120, 120), 255, -1)
cv2.rectangle(rect2, (80, 80), (180, 180), 255, -1)

# Bitwise AND (intersection)
bit_and = cv2.bitwise_and(rect1, rect2)

# Bitwise OR (union)
bit_or = cv2.bitwise_or(rect1, rect2)

# Bitwise XOR (exclusive or)
bit_xor = cv2.bitwise_xor(rect1, rect2)

# Bitwise NOT (inverse)
bit_not = cv2.bitwise_not(rect1)

print("Bitwise operations created: AND, OR, XOR, NOT")


# =============================================================================
# 6. CHANNEL OPERATIONS
# =============================================================================
print("\n--- 6. Channel Operations ---")

# Create color image
color_img = np.zeros((100, 100, 3), dtype=np.uint8)
color_img[:, :, 0] = 255  # Blue channel
color_img[:, :, 1] = 128  # Green channel
color_img[:, :, 2] = 64   # Red channel

# Split into separate channels
b, g, r = cv2.split(color_img) # cv: b,g,r = (100,100) => b (100,100,1) = color_img[:, :, 0] 
print(f"size by using numpy:{color_img[:,:,0].shape}")
print(f"Blue channel shape: {b.shape}")
print(f"Blue value: {b[0, 0]}, Green value: {g[0, 0]}, Red value: {r[0, 0]}")

# Merge channels back
merged = cv2.merge([b, g, r])
print(f"Merged shape: {merged.shape}")

# Swap channels (BGR to RGB)
rgb_img = cv2.merge([r, g, b])

# Or simply:
rgb_img2 = color_img[:, :, ::-1]  # Reverse channel order


# =============================================================================
# 7. REGION OF INTEREST (ROI)
# =============================================================================
print("\n--- 7. Region of Interest (ROI) ---")

# Create image with rectangle
img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)

# Extract ROI
roi = img[50:150, 50:150]
print(f"ROI shape: {roi.shape}")

# Copy ROI to another location
img[100:200, 200:300] = roi
print("ROI copied to new location")


# =============================================================================
# 8. MAKING BORDERS
# =============================================================================
print("\n--- 8. Adding Borders ---")

small_img = np.ones((50, 50, 3), dtype=np.uint8) * 128

# Different border types
border_constant = cv2.copyMakeBorder(small_img, 10, 10, 10, 10,
                                      cv2.BORDER_CONSTANT, value=(255, 0, 0))
border_reflect = cv2.copyMakeBorder(small_img, 10, 10, 10, 10,
                                     cv2.BORDER_REFLECT)
border_replicate = cv2.copyMakeBorder(small_img, 10, 10, 10, 10,
                                       cv2.BORDER_REPLICATE)

print(f"Original: {small_img.shape}, With border: {border_constant.shape}")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display demo images"""
    # Stack images for display
    row1 = np.hstack([black_img, white_img, blue_img])
    row2 = np.hstack([red_img, green_img, cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)])

    display = np.vstack([row1, row2])
    display = cv2.resize(display, (800, 400))

    cv2.imshow("Core Module Demo - Colors", display)

    # Bitwise operations
    bitwise_display = np.hstack([
        cv2.cvtColor(rect1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(rect2, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(bit_and, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(bit_or, cv2.COLOR_GRAY2BGR)
    ])
    bitwise_display = cv2.resize(bitwise_display, (800, 200))
    cv2.putText(bitwise_display, "Rect1 | Rect2 | AND | OR", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Bitwise Operations", bitwise_display)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Demo: Press any key to see visualizations")
    print("=" * 60)
    show_demo()
