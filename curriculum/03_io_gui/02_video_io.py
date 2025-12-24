"""
Module 3: I/O and GUI - Video Capture and Writing
==================================================
Working with video files and camera input.

Official Docs: https://docs.opencv.org/4.x/dd/de7/group__videoio.html

Topics Covered:
1. VideoCapture from File
2. VideoCapture from Camera
3. Video Properties
4. VideoWriter
5. Frame-by-Frame Processing
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 3: Video Capture and Writing")
print("=" * 60)


# =============================================================================
# 1. VIDEO CAPTURE FROM FILE
# =============================================================================
print("\n--- 1. Reading Video Files ---")

# Create a sample video for testing
def create_test_video(filename, frames=60):
    """Create a simple test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (400, 300))

    for i in range(frames):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        # Moving circle
        x = int(50 + (300 * i / frames))
        cv2.circle(frame, (x, 150), 30, (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    print(f"Created test video: {filename}")

create_test_video("test_video.mp4")

# Open video file
cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video file")
else:
    print("Video opened successfully")

    # Read a single frame
    ret, frame = cap.read()
    if ret:
        print(f"Frame shape: {frame.shape}")

cap.release()


# =============================================================================
# 2. VIDEO CAPTURE FROM CAMERA
# =============================================================================
print("\n--- 2. Camera Capture ---")

# Open default camera (index 0)
# cap = cv2.VideoCapture(0)  # Uncomment to test with real camera

# Camera indices:
# 0 = default/built-in camera
# 1, 2, ... = additional cameras

# With specific backend
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow on Windows
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # Video4Linux on Linux

print("Camera backends available:")
print("  cv2.CAP_ANY      - Auto-detect")
print("  cv2.CAP_DSHOW    - DirectShow (Windows)")
print("  cv2.CAP_V4L2     - Video4Linux (Linux)")
print("  cv2.CAP_AVFOUNDATION - AVFoundation (macOS)")


# =============================================================================
# 3. VIDEO PROPERTIES
# =============================================================================
print("\n--- 3. Video Properties ---")

cap = cv2.VideoCapture("test_video.mp4")

# Get properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
codec = int(cap.get(cv2.CAP_PROP_FOURCC))

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Duration: {frame_count/fps:.2f} seconds")

# Common properties:
properties = """
CAP_PROP_POS_MSEC       - Current position in milliseconds
CAP_PROP_POS_FRAMES     - Current frame number
CAP_PROP_FRAME_WIDTH    - Frame width
CAP_PROP_FRAME_HEIGHT   - Frame height
CAP_PROP_FPS            - Frames per second
CAP_PROP_FRAME_COUNT    - Total frame count
CAP_PROP_FOURCC         - Codec code
CAP_PROP_BRIGHTNESS     - Camera brightness
CAP_PROP_CONTRAST       - Camera contrast
CAP_PROP_SATURATION     - Camera saturation
"""
print(properties)

# Set properties (for camera)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Seek to specific frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)  # Go to frame 30
ret, frame = cap.read()
print(f"Read frame at position: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")

cap.release()


# =============================================================================
# 4. VIDEO WRITER
# =============================================================================
print("\n--- 4. Writing Video Files ---")

# Common FourCC codes
fourcc_codes = """
FourCC Codes (codec identifiers):
  'mp4v' - MPEG-4 (good compatibility, .mp4)
  'XVID' - XVID MPEG-4 (.avi)
  'MJPG' - Motion JPEG (.avi, large files)
  'X264' - H.264 (requires codec, .mp4)
  'avc1' - H.264 for Mac (.mp4)
"""
print(fourcc_codes)

# Create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))

# Write frames
for i in range(90):  # 3 seconds at 30fps
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Gradient background
    frame[:, :, 0] = int(255 * i / 90)  # Blue gradient over time

    # Add text
    cv2.putText(frame, f"Frame {i}", (250, 250),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(frame)

out.release()
print("Created output.mp4 with 90 frames")


# =============================================================================
# 5. FRAME-BY-FRAME PROCESSING
# =============================================================================
print("\n--- 5. Frame Processing Loop ---")

def process_video(input_path, output_path):
    """Process video frame by frame."""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error opening {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        # Process frame (example: convert to grayscale and back)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Add frame number
        cv2.putText(processed, f"Processed: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(processed)
        frame_num += 1

    cap.release()
    out.release()
    print(f"Processed {frame_num} frames -> {output_path}")

process_video("test_video.mp4", "processed_video.mp4")


# =============================================================================
# VISUALIZATION
# =============================================================================
def show_demo():
    """Play video with processing."""
    cap = cv2.VideoCapture("test_video.mp4")

    if not cap.isOpened():
        print("Cannot open video")
        return

    print("\nPlaying video... Press 'q' to quit, 'p' to pause")

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Display frame info
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Video Playback", frame)

        key = cv2.waitKey(33) & 0xFF  # ~30 FPS

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Playing")

    cap.release()
    cv2.destroyAllWindows()

    # Cleanup
    import os
    for f in ["test_video.mp4", "output.mp4", "processed_video.mp4"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running video I/O demonstrations...")
    print("=" * 60)
    show_demo()
