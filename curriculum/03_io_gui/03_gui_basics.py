"""
Module 3: I/O and GUI - GUI Basics
==================================
Windows, trackbars, and mouse events.

Official Docs: https://docs.opencv.org/4.x/dc/d4d/tutorial_py_table_of_contents_gui.html

Topics Covered:
1. Window Management
2. Keyboard Input
3. Trackbars (Sliders)
4. Mouse Events
5. Drawing with Mouse
"""

import cv2
import numpy as np

print("=" * 60)
print("Module 3: GUI Basics")
print("=" * 60)


# =============================================================================
# 1. WINDOW MANAGEMENT
# =============================================================================
print("\n--- 1. Window Management ---")

# Create windows with different properties
# cv2.namedWindow("Normal", cv2.WINDOW_NORMAL)       # Resizable
# cv2.namedWindow("AutoSize", cv2.WINDOW_AUTOSIZE)   # Fixed size
# cv2.namedWindow("OpenGL", cv2.WINDOW_OPENGL)       # OpenGL support

# Create sample image
img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (350, 250), (0, 255, 0), 2)
cv2.putText(img, "Window Demo", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

window_info = """
Window flags:
  WINDOW_NORMAL   - Resizable window
  WINDOW_AUTOSIZE - Fixed to image size
  WINDOW_FULLSCREEN - Fullscreen mode
  WINDOW_FREERATIO - Free aspect ratio
  WINDOW_KEEPRATIO - Keep aspect ratio
"""
print(window_info)


# =============================================================================
# 2. KEYBOARD INPUT
# =============================================================================
print("\n--- 2. Keyboard Input ---")

def keyboard_demo():
    """Demonstrate keyboard handling."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    text = "Press keys: q=quit, r=red, g=green, b=blue"

    color = (255, 255, 255)

    while True:
        display = img.copy()
        cv2.rectangle(display, (50, 100), (350, 200), color, -1)
        cv2.putText(display, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, f"Color: {color}", (10, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Keyboard Demo", display)

        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            color = (0, 0, 255)
        elif key == ord('g'):
            color = (0, 255, 0)
        elif key == ord('b'):
            color = (255, 0, 0)

    cv2.destroyAllWindows()


# Key codes for special keys
key_codes = """
Special key handling:
  key = cv2.waitKey(0) & 0xFF

  Arrow keys (may vary by platform):
    Up    = 82 or 0 (with waitKeyEx)
    Down  = 84 or 1
    Left  = 81 or 2
    Right = 83 or 3

  Other:
    ESC   = 27
    Enter = 13
    Space = 32
"""
print(key_codes)


# =============================================================================
# 3. TRACKBARS (SLIDERS)
# =============================================================================
print("\n--- 3. Trackbars ---")

# Trackbar callback function
def on_trackbar(val):
    """Called when trackbar value changes."""
    pass  # We'll read values in the main loop


def trackbar_demo():
    """Demonstrate trackbar controls."""
    cv2.namedWindow("Trackbar Demo")

    # Create trackbars
    cv2.createTrackbar("R", "Trackbar Demo", 0, 255, on_trackbar)
    cv2.createTrackbar("G", "Trackbar Demo", 0, 255, on_trackbar)
    cv2.createTrackbar("B", "Trackbar Demo", 0, 255, on_trackbar)
    cv2.createTrackbar("Size", "Trackbar Demo", 50, 200, on_trackbar)

    print("Adjust trackbars to change color and size. Press 'q' to quit.")

    while True:
        # Get trackbar values
        r = cv2.getTrackbarPos("R", "Trackbar Demo")
        g = cv2.getTrackbarPos("G", "Trackbar Demo")
        b = cv2.getTrackbarPos("B", "Trackbar Demo")
        size = cv2.getTrackbarPos("Size", "Trackbar Demo")

        # Create image with current settings
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(img, (200, 150), size, (b, g, r), -1)

        cv2.imshow("Trackbar Demo", img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# =============================================================================
# 4. MOUSE EVENTS
# =============================================================================
print("\n--- 4. Mouse Events ---")

# Mouse event types
mouse_events = """
Mouse event types:
  EVENT_MOUSEMOVE     - Mouse moved
  EVENT_LBUTTONDOWN   - Left button pressed
  EVENT_RBUTTONDOWN   - Right button pressed
  EVENT_MBUTTONDOWN   - Middle button pressed
  EVENT_LBUTTONUP     - Left button released
  EVENT_RBUTTONUP     - Right button released
  EVENT_LBUTTONDBLCLK - Left double-click
  EVENT_MOUSEWHEEL    - Mouse wheel scroll

Mouse flags (for checking modifier keys):
  EVENT_FLAG_LBUTTON  - Left button is down
  EVENT_FLAG_RBUTTON  - Right button is down
  EVENT_FLAG_CTRLKEY  - Ctrl is pressed
  EVENT_FLAG_SHIFTKEY - Shift is pressed
  EVENT_FLAG_ALTKEY   - Alt is pressed
"""
print(mouse_events)


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    """Handle mouse events."""
    img = param['img']

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left click at ({x}, {y})")
        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right click at ({x}, {y})")
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            # Drawing while left button held
            cv2.circle(img, (x, y), 5, (255, 255, 0), -1)


def mouse_demo():
    """Demonstrate mouse event handling."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(img, "Left click: Green | Right click: Red | Drag: Yellow",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.namedWindow("Mouse Demo")
    cv2.setMouseCallback("Mouse Demo", mouse_callback, {'img': img})

    print("Click and drag on the window. Press 'q' to quit, 'c' to clear.")

    while True:
        cv2.imshow("Mouse Demo", img)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            img[:] = 0
            cv2.putText(img, "Left click: Green | Right click: Red | Drag: Yellow",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.destroyAllWindows()


# =============================================================================
# 5. DRAWING WITH MOUSE
# =============================================================================
print("\n--- 5. Interactive Drawing ---")

class DrawingApp:
    """Simple drawing application."""

    def __init__(self):
        self.drawing = False
        self.mode = 'line'  # 'line', 'rectangle', 'circle'
        self.color = (0, 255, 0)
        self.start_point = None
        self.img = np.zeros((500, 700, 3), dtype=np.uint8)
        self.temp_img = self.img.copy()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_img = self.img.copy()

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_img = self.img.copy()
            if self.mode == 'line':
                cv2.line(self.temp_img, self.start_point, (x, y), self.color, 2)
            elif self.mode == 'rectangle':
                cv2.rectangle(self.temp_img, self.start_point, (x, y), self.color, 2)
            elif self.mode == 'circle':
                radius = int(np.sqrt((x - self.start_point[0])**2 +
                                     (y - self.start_point[1])**2))
                cv2.circle(self.temp_img, self.start_point, radius, self.color, 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == 'line':
                cv2.line(self.img, self.start_point, (x, y), self.color, 2)
            elif self.mode == 'rectangle':
                cv2.rectangle(self.img, self.start_point, (x, y), self.color, 2)
            elif self.mode == 'circle':
                radius = int(np.sqrt((x - self.start_point[0])**2 +
                                     (y - self.start_point[1])**2))
                cv2.circle(self.img, self.start_point, radius, self.color, 2)
            self.temp_img = self.img.copy()

    def run(self):
        cv2.namedWindow("Drawing App")
        cv2.setMouseCallback("Drawing App", self.mouse_callback)

        print("\nDrawing App Controls:")
        print("  l = Line mode")
        print("  r = Rectangle mode")
        print("  c = Circle mode")
        print("  1-3 = Change color (green/red/blue)")
        print("  x = Clear canvas")
        print("  q = Quit")

        while True:
            # Show instructions
            display = self.temp_img.copy()
            cv2.putText(display, f"Mode: {self.mode} | Color: {self.color}",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Drawing App", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('l'):
                self.mode = 'line'
            elif key == ord('r'):
                self.mode = 'rectangle'
            elif key == ord('c'):
                self.mode = 'circle'
            elif key == ord('1'):
                self.color = (0, 255, 0)
            elif key == ord('2'):
                self.color = (0, 0, 255)
            elif key == ord('3'):
                self.color = (255, 0, 0)
            elif key == ord('x'):
                self.img = np.zeros((500, 700, 3), dtype=np.uint8)
                self.temp_img = self.img.copy()

        cv2.destroyAllWindows()


# =============================================================================
# MAIN DEMO
# =============================================================================
def show_demo():
    """Run all GUI demos."""
    print("\n" + "=" * 60)
    print("GUI Demo Menu")
    print("=" * 60)
    print("1. Keyboard demo")
    print("2. Trackbar demo")
    print("3. Mouse demo")
    print("4. Drawing app")
    print("q. Quit")

    while True:
        choice = input("\nEnter choice (1-4 or q): ").strip()

        if choice == '1':
            keyboard_demo()
        elif choice == '2':
            trackbar_demo()
        elif choice == '3':
            mouse_demo()
        elif choice == '4':
            app = DrawingApp()
            app.run()
        elif choice == 'q':
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    show_demo()
