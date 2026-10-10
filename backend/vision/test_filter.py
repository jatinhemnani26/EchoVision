import cv2
import numpy as np

def prepare_for_tesseract(frame: np.ndarray) -> np.ndarray:
    """
    Standard professional OCR preprocessing for webcam frames:
    1. Grayscale
    2. Bilateral filter (smooths noise while keeping text edges razor-sharp)
    3. Otsu thresholding (auto-calculates optimal lighting threshold without shredding thick letters)
    """
    if frame is None:
        return frame
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter preserves sharp letter edges while smoothing webcam sensor noise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Otsu thresholding: automatically finds the global threshold for clean, solid letters
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh
