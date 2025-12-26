"""
Extra Module: Text Detection and OCR
=====================================
Detecting and recognizing text in images.

Topics Covered:
1. Text Detection with EAST
2. Text Recognition Basics
3. Integration with Tesseract
4. EasyOCR Alternative
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for sample_data import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sample_data import get_image

print("=" * 60)
print("Extra Module: Text Detection and OCR")
print("=" * 60)


# =============================================================================
# 1. TEXT DETECTION OVERVIEW
# =============================================================================
print("\n--- 1. Text Detection Overview ---")

overview = """
Text Detection Methods:

1. OpenCV Text Module (opencv-contrib):
   - EAST detector (deep learning)
   - TextDetectorCNN
   - ERFilter (Extremal Region)

2. Traditional Approaches:
   - MSER (Maximally Stable Extremal Regions)
   - Connected component analysis
   - Stroke Width Transform

3. Deep Learning:
   - EAST (Efficient and Accurate Scene Text)
   - CRAFT (Character Region Awareness)
   - DBNet
   - PaddleOCR

Model Files for EAST:
  frozen_east_text_detection.pb
  Download: https://github.com/oyyd/frozen_east_text_detection.pb
"""
print(overview)


# =============================================================================
# 2. CREATE TEST IMAGE WITH TEXT
# =============================================================================
print("\n--- 2. Creating Test Image ---")


def load_text_image():
    """Load a real image with text for OCR demo or create fallback."""
    # Try to load real images with text
    # OpenCV samples: imageTextN.png, imageTextR.png (text), licenseplate_motion.jpg, sudoku.png
    for sample in ["imageTextN.png", "imageTextR.png", "licenseplate_motion.jpg", "sudoku.png", "text_defocus.jpg"]:
        img = get_image(sample)
        if img is not None:
            print(f"Using sample image: {sample}")
            return cv2.resize(img, (600, 400))

    # Fallback: Create image with text
    print("No sample text image found. Using synthetic text image.")
    print("Run: python curriculum/sample_data/download_samples.py")
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240

    # Add text with various styles
    cv2.putText(img, "Hello World!", (50, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    cv2.putText(img, "OpenCV Text", (50, 150),
               cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 150), 2)

    cv2.putText(img, "DETECTION", (50, 220),
               cv2.FONT_HERSHEY_TRIPLEX, 1.5, (150, 0, 0), 2)

    cv2.putText(img, "123 ABC xyz", (50, 290),
               cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 100, 0), 2)

    cv2.putText(img, "Small text here", (50, 350),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    return img


text_img = load_text_image()


# =============================================================================
# 3. MSER TEXT DETECTION
# =============================================================================
print("\n--- 3. MSER Text Detection ---")

# MSER detects stable regions that often correspond to text
gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)

# Create MSER detector
mser = cv2.MSER_create()

# Detect regions
regions, _ = mser.detectRegions(gray)

print(f"MSER regions detected: {len(regions)}")

# Draw bounding boxes for regions
mser_img = text_img.copy()
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(mser_img, hulls, True, (0, 255, 0), 1)

# Filter and merge regions
bboxes = []
for p in regions:
    x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
    aspect_ratio = w / float(h) if h > 0 else 0

    # Filter by aspect ratio and size
    if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:
        bboxes.append((x, y, w, h))

print(f"Filtered regions: {len(bboxes)}")

mser_info = """
MSER Parameters:
  _delta      - Step between threshold levels
  _min_area   - Minimum region area
  _max_area   - Maximum region area
  _max_variation - Max variation in region
  _min_diversity - Min diversity of regions
"""
print(mser_info)


# =============================================================================
# 4. EAST TEXT DETECTOR (CONCEPTUAL)
# =============================================================================
print("\n--- 4. EAST Text Detector ---")

east_example = '''
# EAST Text Detection Example

# Load model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Get output layer names
outputLayers = ["feature_fusion/Conv_7/Sigmoid",  # Scores
                "feature_fusion/concat_3"]         # Geometry

# Prepare image
# EAST requires image dimensions divisible by 32
(H, W) = image.shape[:2]
newW, newH = (320, 320)  # or (640, 640), etc.

blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                             (123.68, 116.78, 103.94),
                             swapRB=True, crop=False)

# Forward pass
net.setInput(blob)
(scores, geometry) = net.forward(outputLayers)

# Decode predictions
# scores: confidence of text at each location
# geometry: rotated bounding box parameters

# Apply NMS to remove duplicates
boxes, confidences = decode_predictions(scores, geometry)
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)

# Draw detected text regions
for i in indices.flatten():
    # Draw rotated rectangle
    vertices = cv2.boxPoints(boxes[i])
    cv2.polylines(image, [vertices.astype(np.int32)], True, (0, 255, 0), 2)
'''
print(east_example)


# =============================================================================
# 5. OCR WITH TESSERACT
# =============================================================================
print("\n--- 5. OCR with Tesseract ---")

tesseract_info = """
Tesseract OCR Integration:

Installation:
  # macOS
  brew install tesseract

  # Ubuntu
  sudo apt install tesseract-ocr

  # Python wrapper
  pip install pytesseract

Usage:
  import pytesseract
  from PIL import Image

  # Simple usage
  text = pytesseract.image_to_string(Image.open('text.png'))

  # With OpenCV
  text = pytesseract.image_to_string(cv2_image)

  # Get bounding boxes
  boxes = pytesseract.image_to_boxes(image)

  # Get detailed data
  data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

Configuration:
  # Custom config
  custom_config = r'--oem 3 --psm 6'
  text = pytesseract.image_to_string(image, config=custom_config)

  # OEM (OCR Engine Mode):
  #   0 = Legacy only
  #   1 = Neural nets LSTM only
  #   2 = Legacy + LSTM
  #   3 = Default

  # PSM (Page Segmentation Mode):
  #   3 = Fully automatic
  #   6 = Assume single block of text
  #   7 = Treat as single line
  #   8 = Treat as single word
  #   10 = Treat as single character
"""
print(tesseract_info)


# =============================================================================
# 6. EASYOCR ALTERNATIVE
# =============================================================================
print("\n--- 6. EasyOCR Alternative ---")

easyocr_info = """
EasyOCR - Simple Deep Learning OCR:

Installation:
  pip install easyocr

Usage:
  import easyocr

  # Create reader (first time downloads models)
  reader = easyocr.Reader(['en'])  # or ['en', 'ch_sim'] for multiple

  # Read text
  result = reader.readtext('image.png')

  # Result format: [(bbox, text, confidence), ...]
  for bbox, text, conf in result:
      print(f"{text} ({conf:.2f})")

  # With OpenCV image
  result = reader.readtext(cv2_image)

  # Detection only
  result = reader.detect('image.png')

  # Recognition only (with given boxes)
  result = reader.recognize('image.png', boxes)

Advantages:
  - Easy to use
  - 80+ languages
  - GPU support
  - Good accuracy out of box

Disadvantages:
  - Slower than Tesseract
  - Larger model files
  - Needs more RAM
"""
print(easyocr_info)


# =============================================================================
# 7. TEXT PREPROCESSING
# =============================================================================
print("\n--- 7. Text Preprocessing ---")


def preprocess_for_ocr(img):
    """Preprocess image for better OCR."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Thresholding
    # Otsu's method
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive threshold (for uneven lighting)
    adaptive = cv2.adaptiveThreshold(denoised, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # Deskew (if needed)
    # coords = np.column_stack(np.where(otsu > 0))
    # angle = cv2.minAreaRect(coords)[-1]
    # Rotate by -angle

    # Rescaling (for small text)
    # scale = 2
    # resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return otsu


preprocessed = preprocess_for_ocr(text_img)

preprocess_tips = """
OCR Preprocessing Tips:

1. Convert to grayscale
2. Remove noise (bilateral filter, NLMeans)
3. Apply thresholding (Otsu, adaptive)
4. Correct skew/rotation
5. Remove borders/lines if needed
6. Upscale small text (2-3x)
7. Invert if dark background
8. Morphological operations to connect text
"""
print(preprocess_tips)


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Display text detection demos."""

    # Original with text
    cv2.imshow("Text Image", text_img)

    # MSER detection
    mser_display = text_img.copy()
    for x, y, w, h in bboxes[:20]:  # Show first 20
        cv2.rectangle(mser_display, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.putText(mser_display, "MSER Detection", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("MSER Detection", mser_display)

    # Preprocessed
    cv2.imshow("Preprocessed for OCR", preprocessed)

    # Grayscale
    cv2.imshow("Grayscale", gray)

    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running text detection demonstrations...")
    print("=" * 60)
    show_demo()
