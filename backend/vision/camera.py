import cv2
import logging
from typing import Optional
import numpy as np
import threading

# Configure basic logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraManager:
    """
    Handles high-performance local frame capture.
    Supports built-in webcams and external sources (like a phone app simulating Ring).
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()

    def start(self) -> bool:
        """Initializes the camera capture."""
        with self.lock:
            if self.capture is not None and self.capture.isOpened():
                return True
                
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) # CAP_DSHOW for faster startup on Windows
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera index {self.camera_index}. Check if it is connected or blocked.")
                return False
            
            # Optimize for low latency (no buffering)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Upgrade to 1080p for crisp small text OCR
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            logger.info(f"Camera {self.camera_index} initialized successfully.")
            return True

    def get_frame(self) -> Optional[np.ndarray]:
        """Reads a frame from the initialized camera in a thread-safe way."""
        with self.lock:
            if not self.capture or not self.capture.isOpened():
                raise RuntimeError("Camera is not initialized. Call start() first.")
            
            ret, frame = self.capture.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                return None
            return frame

    def release(self):
        """Releases the camera resource safely."""
        with self.lock:
            if self.capture:
                self.capture.release()
                self.capture = None
                logger.info(f"Camera {self.camera_index} resources released.")
