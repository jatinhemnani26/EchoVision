from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import cv2
import base64
import numpy as np
from typing import Optional

from vision.camera import CameraManager
from vision.preprocessor import VisionPreprocessor
from vision.ocr import LocalOCREngine
from vision.router import VisionRouter
from vision.cloud_brain import CloudMultimodalEngine

app = FastAPI(
    title="EchoVision Core API",
    description="Local OpenCV Vision Processing, 2-Tier Triage & Alexa MCP Bridge",
    version="1.0.0"
)

# Enable CORS for the local frontend (http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
camera_manager = CameraManager(camera_index=0)
ocr_engine = LocalOCREngine()
vision_router = VisionRouter(ocr_engine=ocr_engine)
cloud_brain = CloudMultimodalEngine()
stream_active = False


class AnalyzeRequest(BaseModel):
    intent: Optional[str] = Field(default="general", description="Voice query or user intent (e.g. 'read this label', 'what is in front of me')")
    camera_index: Optional[int] = Field(default=0, description="0 for laptop webcam, 1 for external phone cam")
    lang: Optional[str] = Field(default="en-US", description="The spoken language code of the user (e.g., hi-IN, es-ES)")
    image_base64: Optional[str] = Field(default=None, description="Direct base64 encoded JPEG from browser camera")


@app.on_event("startup")
def startup_event():
    # Do NOT keep camera on at startup - stay privacy-first and battery friendly!
    # Camera activates strictly on-demand or when user toggles stream ON.
    pass


@app.on_event("shutdown")
def shutdown_event():
    global stream_active
    stream_active = False
    camera_manager.release()


@app.post("/api/v1/vision/camera/stop")
def stop_camera_hardware():
    """Explicitly releases and powers down the physical webcam hardware sensor (turns off LED)."""
    global stream_active
    stream_active = False
    camera_manager.release()
    return {"status": "camera_powered_off", "camera_ready": False}


@app.get("/health")
def health_check():
    """Health check endpoint for MCP gateway and frontend dashboard."""
    return {
        "status": "online",
        "service": "EchoVision Core Engine",
        "ocr_available": ocr_engine.is_available,
        "camera_ready": camera_manager.capture is not None and camera_manager.capture.isOpened()
    }


from fastapi.responses import StreamingResponse
import time

def generate_frames(cam_index: int = 0):
    """Generator for MJPEG stream from the local hardware."""
    global stream_active
    stream_active = True

    if camera_manager.camera_index != cam_index or not camera_manager.capture or not camera_manager.capture.isOpened():
        camera_manager.release()
        camera_manager.camera_index = cam_index
        camera_manager.start()

    try:
        while stream_active:
            frame = camera_manager.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
                
            # Compress for stream speed (Restored fast 1080p stream without blocking OpenCV overlays)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        # Client disconnected or toggled off -> shut off camera hardware completely
        stream_active = False
        camera_manager.release()

@app.post("/api/v1/alexa")
async def alexa_webhook(req: dict):
    """
    Official Alexa Custom Skill Webhook.
    Receives an IntentRequest from Amazon Cloud, triggers the edge camera, analyzes via NVIDIA NIM, 
    and returns Alexa-formatted SSML speech.
    """
    request_type = req.get("request", {}).get("type")
    
    # Handle Alexa Skill launch or Intent
    if request_type == "LaunchRequest" or request_type == "IntentRequest":
        was_streaming = stream_active
        if not camera_manager.capture or not camera_manager.capture.isOpened():
            camera_manager.start()

        frame = camera_manager.get_frame()

        if not was_streaming:
            camera_manager.release()

        if frame is None:
            spoken_text = "I'm sorry, my camera sensor is currently offline."
        else:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Analyze using NVIDIA NIM
            cloud_result = cloud_brain.analyze_image(base64_image, "Describe the primary object or scene in a concise sentence suitable for audio playback. Read any visible text.")
            if cloud_result["success"]:
                spoken_text = cloud_result["description"]
            else:
                spoken_text = "I encountered an error analyzing the environment."

        # Return official Alexa JSON Format
        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": spoken_text
                },
                "shouldEndSession": True
            }
        }
    return {}


@app.get("/api/v1/vision/stream")
def video_feed(cam_index: int = 0):
    """MJPEG stream endpoint for the Atmos Gallery Frontend HUD."""
    return StreamingResponse(generate_frames(cam_index), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/v1/vision/analyze")
def analyze_scene(req: AnalyzeRequest):
    """
    Core 2-Tier Triage Pipeline:
    1. If client sent an image from browser camera (image_base64), decode it directly.
    2. Otherwise, grabs live frame from hardware (starting camera on-demand if needed).
    3. Runs local OCR + NVIDIA VLM analysis.
    """
    frame = None
    if req.image_base64:
        try:
            clean_b64 = req.image_base64
            if "," in clean_b64:
                clean_b64 = clean_b64.split(",")[1]
            img_bytes = base64.b64decode(clean_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")

    if frame is None:
        was_streaming = stream_active
        if not camera_manager.capture or not camera_manager.capture.isOpened():
            camera_manager.camera_index = req.camera_index
            if not camera_manager.start():
                raise HTTPException(status_code=500, detail=f"Failed to access camera index {req.camera_index}.")

        frame = camera_manager.get_frame()

        # Power down camera sensor immediately if user hasn't enabled continuous live feed
        if not was_streaming:
            camera_manager.release()

    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from hardware or browser.")

    # Step 1: Document / Edge Preprocessing for Local OCR ONLY
    ocr_preprocessed = VisionPreprocessor.optimize_for_ocr(frame)

    # Step 2: Evaluate via 2-Tier Cost-Mitigation Router
    routing_decision = vision_router.evaluate_frame(ocr_preprocessed, user_intent=req.intent)

    # Step 3: For the Cloud Brain (NVIDIA VLM), use the full natural COLOR frame!
    # VLMs need true RGB colors, textures, and lighting to read text and identify objects.
    # Using 85 Quality to keep payload size reasonable while keeping 1080p resolution
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    base64_image = base64.b64encode(buffer).decode('utf-8')

    response_payload = {
        "status": "success",
        "intent": req.intent,
        "routing": routing_decision,
        "frame_meta": {
            "resolution": [int(frame.shape[1]), int(frame.shape[0])]
        }
    }

    # Only include full base64 image data and call cloud if needed
    if routing_decision["needs_cloud"]:
        response_payload["cloud_payload"] = {
            "image_format": "jpeg"
        }
        # Run Phase 3: The NVIDIA Cloud Brain on the REAL COLOR frame
        cloud_result = cloud_brain.analyze_image(base64_image, req.intent, lang=req.lang)
        response_payload["cloud_analysis"] = cloud_result

    return JSONResponse(content=response_payload)


@app.get("/api/v1/vision/ocr")
def read_text_direct():
    """Direct local OCR execution on current frame."""
    frame = camera_manager.get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame.")

    enhanced = VisionPreprocessor.optimize_for_ocr(frame)
    result = ocr_engine.extract_text(enhanced)
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
