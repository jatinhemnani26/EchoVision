import logging
from typing import Dict, Any
import numpy as np
from vision.ocr import LocalOCREngine

logger = logging.getLogger(__name__)

class VisionRouter:
    """
    Intelligent 2-Tier Triage Layer (EchoVision Cost-Mitigation Engine).
    Determines whether a user query can be satisfied purely via local edge OCR (Free)
    or requires Cloud Multimodal Reasoning (NVIDIA Nemotron via Nebius Cloud).
    """

    def __init__(self, ocr_engine: LocalOCREngine):
        self.ocr_engine = ocr_engine

    def evaluate_frame(self, enhanced_frame: np.ndarray, user_intent: str = "general") -> Dict[str, Any]:
        """
        Analyzes the frame and user intent to determine routing path.
        
        Routing rules:
        1. If user intent is explicit reading ("read this", "what does the label say"):
           - Try local OCR first. If word count >= 3 and confidence >= 50, resolve locally (Tier 1: Free).
        2. If user intent is spatial / object recognition ("what is in front of me", "find my keys"):
           - Route directly to Cloud Multimodal Vision (Tier 2).
        3. Automatic fallback: If local OCR yields gibberish or empty text, escalate to Cloud.
        """
        ocr_result = self.ocr_engine.extract_text(enhanced_frame)
        
        has_legible_text = (
            ocr_result["success"] 
            and ocr_result["word_count"] >= 3 
            and ocr_result["confidence"] >= 50.0
        )

        reading_intents = ["read", "text", "label", "document", "letter", "medicine", "expiry", "receipt"]
        is_reading_query = any(keyword in user_intent.lower() for keyword in reading_intents)

        if is_reading_query and has_legible_text:
            return {
                "route": "LOCAL_EDGE",
                "tier": 1,
                "cost_incurred": False,
                "reason": "Clear text detected for reading query. Local OCR fulfilled request.",
                "ocr_data": ocr_result,
                "needs_cloud": False
            }
        
        return {
            "route": "CLOUD_MULTIMODAL",
            "tier": 2,
            "cost_incurred": True,
            "reason": "Complex spatial reasoning, navigation, or low OCR confidence requires NVIDIA Multimodal Vision.",
            "ocr_data": ocr_result,
            "needs_cloud": True
        }
