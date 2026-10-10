import os
import logging
from dotenv import load_dotenv

# Load credentials from .env safely
load_dotenv()

logger = logging.getLogger(__name__)

class CloudMultimodalEngine:
    """
    Phase 3: The Cloud VLM Brain.
    Connects to NVIDIA NIM (build.nvidia.com) & Nebius AI endpoints.
    Reads credentials strictly from environment variables (.env).
    """
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = os.getenv("NVIDIA_VISION_MODEL", "meta/llama-3.2-11b-vision-instruct")
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                # NVIDIA NIM exposes an OpenAI-compatible endpoint
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
                logger.info(f"Connected to NVIDIA NIM Cloud Brain ({self.model})")
            except ImportError:
                logger.error("OpenAI SDK not found. Run: pip install openai")
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIA NIM client: {e}")
        else:
            logger.warning("No NVIDIA_API_KEY found in .env. Running in mock simulation mode.")

    def analyze_image(self, base64_image: str, intent: str, lang: str = "en-US") -> dict:
        """
        Sends the optimized OpenCV frame to NVIDIA Llama-3.2-Vision NIM.
        Returns a concise, spatial, and accurate natural language description for the user.
        """
        if not self.client or not self.api_key:
            return {
                "success": False,
                "description": "NVIDIA API key not configured in .env file.",
                "model_used": "none"
            }

        if lang == "hi-IN":
            lang_rules = (
                "LANGUAGE MANDATE: You MUST reply in HINDI using standard DEVANAGARI SCRIPT (हिन्दी लिपि). "
                "Example: 'आपके सामने एक लाल रंग की वस्तु है।' "
                "DO NOT use Latin/English alphabet (do NOT write Hinglish like 'Aapke saamne'). "
                "Write ONLY in pure Devanagari Hindi characters."
            )
        elif lang == "es-ES":
            lang_rules = "LANGUAGE MANDATE: You MUST reply completely in SPANISH (Español). Do not speak English."
        elif lang == "fr-FR":
            lang_rules = "LANGUAGE MANDATE: You MUST reply completely in FRENCH (Français). Do not speak English."
        else:
            lang_rules = "LANGUAGE MANDATE: Reply in clear, concise English."

        prompt_text = (
            f"You are EchoVision, an ambient visual assistant for a visually impaired user.\n"
            f"{lang_rules}\n"
            f"The user says: '{intent}'.\n"
            f"SECURITY CONSTRAINT: You MUST only answer questions related to the provided image, "
            f"the user's physical surroundings, or visual accessibility. If the user asks for code generation, "
            f"general knowledge, math, or anything off-topic, you MUST politely refuse and state your purpose as a visual assistant.\n"
            f"Analyze the image and answer the user directly in 1-2 clear, natural, and helpful sentences. "
            f"If there is written text or a document/label, read the key words clearly.\n"
            f"{lang_rules}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.2,
                top_p=0.7
            )

            description = response.choices[0].message.content.strip()
            return {
                "success": True,
                "description": description,
                "model_used": self.model
            }

        except Exception as e:
            logger.error(f"NVIDIA NIM Vision Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "description": "Visual analysis could not be completed by NVIDIA cloud."
            }
