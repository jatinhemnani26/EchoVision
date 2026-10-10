import os
import shutil
import logging
from typing import Dict, Any
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = logging.getLogger(__name__)

class LocalOCREngine:
    """
    High-performance Tesseract OCR Engine configured for solid letter detection.
    """

    def __init__(self):
        self._configure_tesseract_path()
        self.is_available = self._check_availability()

    def _configure_tesseract_path(self):
        """Auto-detect common Windows installation paths for Tesseract OCR."""
        if pytesseract is None:
            return

        if shutil.which("tesseract"):
            return

        common_windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]

        for path in common_windows_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Tesseract binary found at: {path}")
                return

    def _check_availability(self) -> bool:
        if pytesseract is None:
            return False
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialized (v{version}).")
            return True
        except Exception:
            return False

    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_available or pytesseract is None:
            return {
                "success": False,
                "text": "",
                "word_count": 0,
                "confidence": 0.0,
                "error": "Tesseract binary not installed on host."
            }

        try:
            # PSM 3: Fully automatic page segmentation without OSD
            custom_config = r'--oem 3 --psm 3'
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                # Keep real words with decent confidence
                if len(word) > 1 and conf > 30.0:
                    words.append(word)
                    confidences.append(conf)

            full_text = " ".join(words)
            avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

            return {
                "success": True,
                "text": full_text,
                "word_count": len(words),
                "confidence": round(avg_conf, 2),
                "error": None
            }
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "success": False,
                "text": "",
                "word_count": 0,
                "confidence": 0.0,
                "error": str(e)
            }
