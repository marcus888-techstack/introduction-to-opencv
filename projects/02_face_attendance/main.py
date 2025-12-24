"""
Project 2: Automated Attendance System with Face Recognition
=============================================================
A contactless attendance system using face recognition - widely used
in offices and schools.

Key Concepts:
- Face detection (Haar cascades or DNN)
- Face encoding and embedding extraction
- Face matching and comparison
- Database management for registered faces
- Attendance logging with timestamps

Official OpenCV References:
- Face Detection: https://docs.opencv.org/4.x/d2/d64/tutorial_table_of_content_objdetect.html
- DNN Module: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from pathlib import Path

# Try to import face_recognition library
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("face_recognition not installed. Using OpenCV Haar Cascade for detection only.")
    print("Install with: pip install face_recognition")

# Try to import pandas for attendance export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class FaceDatabase:
    """Database to store and manage registered faces."""

    def __init__(self, db_path="face_database"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        self.encodings_file = self.db_path / "encodings.pkl"
        self.known_encodings = []
        self.known_names = []

        self._load_database()

    def _load_database(self):
        """Load existing face encodings from disk."""
        if self.encodings_file.exists():
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data.get('encodings', [])
                self.known_names = data.get('names', [])
            print(f"Loaded {len(self.known_names)} registered faces")

    def _save_database(self):
        """Save face encodings to disk."""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.known_encodings,
                'names': self.known_names
            }, f)

    def register_face(self, name, image):
        """
        Register a new face in the database.

        Args:
            name: Person's name/ID
            image: BGR image containing the face

        Returns:
            True if registration successful, False otherwise
        """
        if not FACE_RECOGNITION_AVAILABLE:
            print("face_recognition library required for registration")
            return False

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)

        if len(face_locations) == 0:
            print("No face detected in image")
            return False

        if len(face_locations) > 1:
            print("Multiple faces detected. Please use an image with single face.")
            return False

        # Get face encoding
        encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]

        # Save face image
        face_img_path = self.db_path / f"{name}.jpg"
        cv2.imwrite(str(face_img_path), image)

        # Add to database
        self.known_encodings.append(encoding)
        self.known_names.append(name)
        self._save_database()

        print(f"Registered: {name}")
        return True

    def recognize_face(self, image, tolerance=0.6):
        """
        Recognize faces in an image.

        Args:
            image: BGR image
            tolerance: How strict the matching should be (lower = stricter)

        Returns:
            List of (name, location, confidence) tuples
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return self._detect_faces_opencv(image)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces and get encodings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            confidence = 0.0

            if len(self.known_encodings) > 0:
                # Compare with known faces
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_match_idx = np.argmin(distances)

                if distances[best_match_idx] < tolerance:
                    name = self.known_names[best_match_idx]
                    confidence = 1.0 - distances[best_match_idx]

            # Convert location format: (top, right, bottom, left) -> (x, y, w, h)
            top, right, bottom, left = location
            results.append((name, (left, top, right - left, bottom - top), confidence))

        return results

    def _detect_faces_opencv(self, image):
        """Fallback face detection using OpenCV Haar Cascade."""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        return [("Unknown", tuple(face), 0.0) for face in faces]

    def get_registered_count(self):
        """Return number of registered faces."""
        return len(self.known_names)

    def list_registered(self):
        """List all registered names."""
        return list(set(self.known_names))


class AttendanceSystem:
    """Main attendance system with logging."""

    def __init__(self, db_path="face_database", log_path="attendance_logs"):
        self.face_db = FaceDatabase(db_path)
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok=True)

        # Track today's attendance
        self.today_attendance = set()
        self._load_today_log()

    def _get_today_log_file(self):
        """Get path to today's attendance log."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_path / f"attendance_{today}.csv"

    def _load_today_log(self):
        """Load already marked attendance for today."""
        log_file = self._get_today_log_file()
        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        self.today_attendance.add(parts[0])

    def mark_attendance(self, name):
        """
        Mark attendance for a person.

        Returns:
            True if newly marked, False if already marked today
        """
        if name in self.today_attendance or name == "Unknown":
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = self._get_today_log_file()

        # Write to CSV
        with open(log_file, 'a') as f:
            f.write(f"{name},{timestamp}\n")

        self.today_attendance.add(name)
        print(f"Attendance marked: {name} at {timestamp}")
        return True

    def get_today_attendance(self):
        """Get list of people who attended today."""
        return list(self.today_attendance)

    def export_attendance(self, output_path=None):
        """Export attendance to Excel/CSV."""
        if not PANDAS_AVAILABLE:
            print("pandas required for export. Install with: pip install pandas")
            return None

        log_file = self._get_today_log_file()
        if not log_file.exists():
            print("No attendance records for today")
            return None

        df = pd.read_csv(log_file, names=['Name', 'Timestamp'])

        if output_path is None:
            output_path = self.log_path / f"attendance_{datetime.now().strftime('%Y-%m-%d')}.xlsx"

        df.to_excel(str(output_path), index=False)
        print(f"Exported to: {output_path}")
        return output_path


def draw_face_box(image, name, location, confidence):
    """Draw bounding box and label for detected face."""
    x, y, w, h = location

    # Color based on recognition status
    if name == "Unknown":
        color = (0, 0, 255)  # Red
    else:
        color = (0, 255, 0)  # Green

    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Draw label background
    label = f"{name}"
    if confidence > 0:
        label += f" ({confidence:.0%})"

    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x, y - label_h - 10), (x + label_w, y), color, -1)

    # Draw label text
    cv2.putText(image, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def run_registration_mode(system):
    """Interactive face registration from webcam."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("\n=== Face Registration Mode ===")
    print("Controls:")
    print("  SPACE - Capture face for registration")
    print("  Q     - Quit registration")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, "Position face and press SPACE to register",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Registration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            name = input("Enter name for this person: ").strip()
            if name:
                success = system.face_db.register_face(name, frame)
                if success:
                    print(f"Successfully registered: {name}")
                else:
                    print("Registration failed. Try again.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_attendance_mode(system):
    """Main attendance marking from webcam."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("\n=== Attendance Mode ===")
    print(f"Registered faces: {system.face_db.get_registered_count()}")
    print(f"Today's attendance: {len(system.today_attendance)}")
    print("Controls:")
    print("  Q - Quit")

    # Process every N frames for performance
    frame_count = 0
    process_every = 3
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Process face recognition
        if frame_count % process_every == 0:
            last_results = system.face_db.recognize_face(frame)

            # Mark attendance for recognized faces
            for name, location, confidence in last_results:
                if name != "Unknown" and confidence > 0.5:
                    newly_marked = system.mark_attendance(name)
                    if newly_marked:
                        print(f"New attendance: {name}")

        # Draw results
        for name, location, confidence in last_results:
            display = draw_face_box(display, name, location, confidence)

        # Show status
        status = f"Registered: {system.face_db.get_registered_count()} | "
        status += f"Present Today: {len(system.today_attendance)}"
        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Attendance System", display)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    print("\n=== Attendance Summary ===")
    print(f"Total present: {len(system.today_attendance)}")
    for name in system.get_today_attendance():
        print(f"  - {name}")


def run_demo_mode():
    """Run demo with synthetic data."""
    print("\n=== Demo Mode ===")
    print("This demo shows the system structure without real face recognition.")

    # Create demo system
    system = AttendanceSystem(db_path="demo_faces", log_path="demo_logs")

    # Simulate registrations
    demo_names = ["Alice", "Bob", "Charlie"]
    print(f"\nSimulated registrations: {demo_names}")

    # Simulate attendance
    print("\nSimulated attendance marking:")
    for name in demo_names[:2]:  # Only first two "show up"
        system.today_attendance.add(name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {name} marked at {timestamp}")

    # Show summary
    print(f"\n=== Summary ===")
    print(f"Total registered: {len(demo_names)}")
    print(f"Present today: {len(system.today_attendance)}")
    print(f"Absent: {set(demo_names) - system.today_attendance}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument("--register", action="store_true", help="Registration mode")
    parser.add_argument("--attendance", action="store_true", help="Attendance mode")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--export", action="store_true", help="Export today's attendance")
    parser.add_argument("--list", action="store_true", help="List registered faces")

    args = parser.parse_args()

    system = AttendanceSystem()

    if args.demo:
        run_demo_mode()
    elif args.register:
        if not FACE_RECOGNITION_AVAILABLE:
            print("face_recognition library required for registration")
            print("Install with: pip install face_recognition")
        else:
            run_registration_mode(system)
    elif args.attendance:
        run_attendance_mode(system)
    elif args.export:
        system.export_attendance()
    elif args.list:
        registered = system.face_db.list_registered()
        print(f"Registered faces ({len(registered)}):")
        for name in registered:
            print(f"  - {name}")
    else:
        print("Face Recognition Attendance System")
        print("=" * 40)
        print(f"Registered faces: {system.face_db.get_registered_count()}")
        print(f"Today's attendance: {len(system.today_attendance)}")
        print()
        print("Usage:")
        print("  python main.py --register    # Register new faces")
        print("  python main.py --attendance  # Start attendance marking")
        print("  python main.py --export      # Export to Excel")
        print("  python main.py --list        # List registered faces")
        print("  python main.py --demo        # Run demo mode")
