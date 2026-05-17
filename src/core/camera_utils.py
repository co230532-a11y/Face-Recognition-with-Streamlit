"""
Camera Utilities - Handle camera capture and face detection
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import sys

# Add parent to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    CAPTURED_DIR, FACE_CASCADE_PATH, LOGS_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'camera.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CameraCapture:
    """Handle camera operations and face detection"""
    
    def __init__(self):
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.detected_faces = []
        
    def initialize_camera(self):
        """Initialize camera connection"""
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")
            
            logger.info(f"Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture single frame from camera"""
        if self.cap is None:
            logger.error("Camera not initialized")
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None
        
        return frame
    
    def detect_faces(self, frame):
        """Detect faces in frame using Haar Cascade"""
        if frame is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def save_face_region(self, frame, face, filename, person_name="captured"):
        """Extract and save face region"""
        if frame is None or len(face) == 0:
            logger.warning("Cannot save face region: invalid input")
            return None
        
        try:
            x, y, w, h = face
            face_region = frame[y:y+h, x:x+w]
            
            # Create person directory
            person_dir = CAPTURED_DIR / person_name
            person_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with timestamp
            filepath = person_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filepath), face_region)
            
            logger.info(f"Face saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save face region: {e}")
            return None
    
    def capture_multiple_frames(self, count=5, delay=500, person_name="captured"):
        """Capture multiple frames for a person"""
        if self.cap is None:
            logger.error("Camera not initialized")
            return []
        
        captured_paths = []
        frames_captured = 0
        
        logger.info(f"Starting capture for '{person_name}' - {count} frames")
        
        while frames_captured < count:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture frame in sequence")
                break
            
            # Display frame with countdown
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing... {frames_captured + 1}/{count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Capture', display_frame)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logger.info("Capture interrupted by user")
                break
            
            # Detect and save face
            faces = self.detect_faces(frame)
            if len(faces) > 0:
                filepath = self.save_face_region(frame, faces[0], f"frame_{frames_captured}")
                if filepath:
                    captured_paths.append(filepath)
                    frames_captured += 1
            else:
                logger.warning(f"No face detected in frame {frames_captured + 1}")
        
        cv2.destroyAllWindows()
        logger.info(f"Capture completed: {len(captured_paths)} frames saved")
        return captured_paths
    
    def live_preview(self, duration_seconds=10):
        """Show live camera preview"""
        if self.cap is None:
            logger.error("Camera not initialized")
            return
        
        logger.info(f"Starting live preview for {duration_seconds} seconds")
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < duration_seconds:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Detect faces and draw rectangles
            faces = self.detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('Live Preview', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        logger.info("Live preview ended")
    
    def release_camera(self):
        """Release camera resource"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released")
    
    def __del__(self):
        self.release_camera()


def capture_database_image(person_name, num_samples=5):
    """Utility function to capture reference database images"""
    camera = CameraCapture()
    if not camera.initialize_camera():
        return []
    
    print(f"\nCapturing database images for: {person_name}")
    print(f"Position your face in front of the camera")
    print(f"Press 'q' to cancel, or wait for automatic capture\n")
    
    paths = camera.capture_multiple_frames(count=num_samples, person_name=person_name)
    camera.release_camera()
    
    return paths


def capture_test_image(person_name="test_subject"):
    """Utility function to capture test image"""
    camera = CameraCapture()
    if not camera.initialize_camera():
        return None
    
    print(f"\nCapturing test image")
    print(f"Position your face and press any key to capture")
    print(f"Press 'q' to cancel\n")
    
    camera.live_preview(duration_seconds=30)
    
    # Capture one final image
    ret, frame = camera.cap.read()
    if ret:
        faces = camera.detect_faces(frame)
        if len(faces) > 0:
            filepath = camera.save_face_region(frame, faces[0], "test_capture", person_name)
            camera.release_camera()
            return filepath
    
    camera.release_camera()
    return None


if __name__ == "__main__":
    # Test camera
    camera = CameraCapture()
    if camera.initialize_camera():
        print("Testing live preview...")
        camera.live_preview(duration_seconds=5)
        camera.release_camera()
        print("Camera test completed")
