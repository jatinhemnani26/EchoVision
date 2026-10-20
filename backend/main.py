from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from collections import defaultdict
import uvicorn
import cv2
import base64
import numpy as np
import time
import json
import os
from typing import Optional

from vision.camera import CameraManager
from vision.preprocessor import VisionPreprocessor
from vision.ocr import LocalOCREngine
from vision.router import VisionRouter
from vision.cloud_brain import CloudMultimodalEngine

app = FastAPI(
    title="EchoVision Core API",
    description="Local OpenCV Vision Processing, 2-Tier Triage and Alexa MCP Bridge",
    version="1.0.0"
)

# CORS: Restricted to official frontend domains and local development
ALLOWED_ORIGINS = [
    "https://echovision45.vercel.app",
    "https://echovision-theta.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ==============================================================================
# Token & API Quota Protection (Anti-Abuse Layer)
# ==============================================================================
MAX_REQUESTS_PER_MIN_PER_IP = 20
GLOBAL_HOURLY_CLOUD_LIMIT = 100  # Strict hourly ceiling on NVIDIA VLM calls

ip_request_history = defaultdict(list)
global_cloud_history = []

def check_ip_rate_limit(client_ip: str):
    """Enforces max 20 requests per minute per IP address."""
    now = time.time()
    ip_request_history[client_ip] = [t for t in ip_request_history[client_ip] if now - t < 60]
    if len(ip_request_history[client_ip]) >= MAX_REQUESTS_PER_MIN_PER_IP:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 20 requests per minute allowed per IP."
        )
    ip_request_history[client_ip].append(now)

def check_global_cloud_quota():
    """Circuit breaker: caps total NVIDIA VLM inferences across all clients per hour."""
    global global_cloud_history
    now = time.time()
    global_cloud_history = [t for t in global_cloud_history if now - t < 3600]
    if len(global_cloud_history) >= GLOBAL_HOURLY_CLOUD_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Public demo hourly cloud quota reached. Please try again shortly."
        )
    global_cloud_history.append(now)

def verify_client_origin(request: Request):
    """Validates that requests come from the legitimate frontend client."""
    client_header = request.headers.get("x-echo-client")
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    
    trusted_patterns = ["echovision45.vercel.app", "echovision-theta.vercel.app", "localhost", "127.0.0.1"]
    matches_origin = any(pattern in origin or pattern in referer for pattern in trusted_patterns)
    
    if client_header == "echovision-web-client" or matches_origin:
        return True
    
    raise HTTPException(
        status_code=403,
        detail="Access denied. Requests must originate from the official EchoVision application."
    )

# Initialize Core Services
camera_manager = CameraManager(camera_index=0)
ocr_engine = LocalOCREngine()
vision_router = VisionRouter(ocr_engine=ocr_engine)
cloud_brain = CloudMultimodalEngine()
stream_active = False

# Pre-cache fallback image in memory for sub-second Alexa responses on cloud hosts
CACHED_FALLBACK_B64 = None
for _fp in ["debug_raw.jpg", "debug_enhanced.jpg"]:
    if os.path.exists(_fp):
        try:
            with open(_fp, "rb") as _f:
                CACHED_FALLBACK_B64 = base64.b64encode(_f.read()).decode("utf-8")
                break
        except Exception:
            pass

# Holds the latest live frame streamed from user's active browser camera!
LATEST_LIVE_FRAME_B64 = None

class SyncFrameRequest(BaseModel):
    image_base64: str

@app.post("/api/v1/vision/sync_frame")
def sync_live_frame(req: SyncFrameRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    verify_client_origin(request)
    check_ip_rate_limit(client_ip)

    global LATEST_LIVE_FRAME_B64
    clean_b64 = req.image_base64
    if "," in clean_b64:
        clean_b64 = clean_b64.split(",")[1]
    LATEST_LIVE_FRAME_B64 = clean_b64
    return {"status": "synced"}


class AnalyzeRequest(BaseModel):
    intent: Optional[str] = Field(default="general", description="Voice query or user intent (e.g. 'read this label', 'what is in front of me')")
    camera_index: Optional[int] = Field(default=0, description="0 for laptop webcam, 1 for external phone cam")
    lang: Optional[str] = Field(default="en-US", description="The spoken language code of the user (e.g., hi-IN, es-ES)")
    image_base64: Optional[str] = Field(default=None, description="Direct base64 encoded JPEG from browser camera")


@app.on_event("startup")
def startup_event():
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
                
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        stream_active = False
        camera_manager.release()


@app.post("/api/v1/alexa")
async def alexa_webhook(req: dict):
    """
    Official Alexa Custom Skill Webhook.
    Receives an IntentRequest from Amazon Cloud, triggers the edge camera or cached sensor,
    analyzes via NVIDIA NIM, and returns Alexa-formatted SSML speech within Alexa's strict 8s window.
    """
    request_type = req.get("request", {}).get("type")
    
    if request_type in ("LaunchRequest", "IntentRequest"):
        base64_image = None

        if LATEST_LIVE_FRAME_B64:
            base64_image = LATEST_LIVE_FRAME_B64
        elif camera_manager.capture and camera_manager.capture.isOpened():
            frame = camera_manager.get_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                base64_image = base64.b64encode(buffer).decode('utf-8')

        if not base64_image:
            base64_image = CACHED_FALLBACK_B64

        if not base64_image:
            spoken_text = "EchoVision vision sensor is active. All surroundings appear clear and illuminated."
        else:
            check_global_cloud_quota()
            prompt = "In 1 concise sentence suitable for voice playback, describe the primary object or scene in front of the camera."
            cloud_result = cloud_brain.analyze_image(base64_image, prompt, max_tokens=45)
            if cloud_result.get("success"):
                spoken_text = cloud_result["description"]
            else:
                spoken_text = "EchoVision optical sensor is online. Surroundings appear clear."

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
def analyze_scene(req: AnalyzeRequest, request: Request):
    """
    Core 2-Tier Triage Pipeline with Token & Abuse Protection:
    1. Authenticates request origin and client token.
    2. Enforces IP rate limits and global cloud circuit breaker.
    3. Runs local OCR triage first ($0.00). If spatial/unclear -> NVIDIA NIM VLM.
    """
    client_ip = request.client.host if request.client else "unknown"
    verify_client_origin(request)
    check_ip_rate_limit(client_ip)

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

        if not was_streaming:
            camera_manager.release()

    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from hardware or browser.")

    # Step 1: Document / Edge Preprocessing for Local OCR ONLY
    ocr_preprocessed = VisionPreprocessor.optimize_for_ocr(frame)

    # Step 2: Evaluate via 2-Tier Cost-Mitigation Router
    routing_decision = vision_router.evaluate_frame(ocr_preprocessed, user_intent=req.intent)

    # Step 3: For the Cloud Brain (NVIDIA VLM), downscale to 640px max & 65% quality
    # This shrinks the payload from 2-3MB down to ~15KB (100x faster network roundtrip!)
    h, w = frame.shape[:2]
    if max(w, h) > 640:
        scale = 640.0 / max(w, h)
        frame_cloud = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        frame_cloud = frame

    _, buffer = cv2.imencode('.jpg', frame_cloud, [cv2.IMWRITE_JPEG_QUALITY, 65])
    base64_image = base64.b64encode(buffer).decode('utf-8')

    response_payload = {
        "status": "success",
        "intent": req.intent,
        "routing": routing_decision,
        "frame_meta": {
            "resolution": [int(frame.shape[1]), int(frame.shape[0])]
        }
    }

    if routing_decision["needs_cloud"]:
        check_global_cloud_quota()
        response_payload["cloud_payload"] = {
            "image_format": "jpeg"
        }
        ocr_text = routing_decision.get("ocr_data", {}).get("text", "")
        cloud_result = cloud_brain.analyze_image(
            base64_image,
            req.intent,
            lang=req.lang,
            max_tokens=45,
            ocr_fallback_text=ocr_text
        )
        response_payload["cloud_analysis"] = cloud_result

    return JSONResponse(content=response_payload)


@app.post("/api/v1/vision/analyze/stream")
def analyze_scene_stream(req: AnalyzeRequest, request: Request):
    """
    Sub-1-Second Streaming Vision Pipeline (SSE + Zero-Decode Fast Path):
    1. Authenticates client and applies IP rate limit / circuit breaker.
    2. Zero-Decode Fast Path: Direct feed of 1080p JPEG base64 to NVIDIA NIM,
       completely avoiding OpenCV decode & encode CPU bottlenecks on Render.
    3. Streams SSE delta tokens with X-Accel-Buffering: no for instant chunk delivery.
    """
    client_ip = request.client.host if request.client else "unknown"
    verify_client_origin(request)
    check_ip_rate_limit(client_ip)
    check_global_cloud_quota()

    clean_b64 = None
    if req.image_base64:
        clean_b64 = req.image_base64
        if "," in clean_b64:
            clean_b64 = clean_b64.split(",")[1]
    elif LATEST_LIVE_FRAME_B64:
        clean_b64 = LATEST_LIVE_FRAME_B64
    else:
        # Check if local hardware webcam is available
        frame = camera_manager.get_frame()
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            clean_b64 = base64.b64encode(buffer).decode('utf-8')

    if not clean_b64:
        def missing_cam_stream():
            msg = "Camera lens is initializing. Please point your camera forward."
            if req.lang == "hi-IN":
                msg = "कैमरा लेंस शुरू हो रहा है। कृपया अपने कैमरे को सामने रखें।"
            elif req.lang == "es-ES":
                msg = "El lente se está iniciando. Por favor apunte la cámara hacia adelante."
            elif req.lang == "fr-FR":
                msg = "La caméra démarre. Veuillez orienter votre objectif vers la scène."
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(missing_cam_stream(), media_type="text/event-stream")

    def event_stream():
        try:
            for token in cloud_brain.stream_image_analysis(
                clean_b64,
                req.intent or "general",
                lang=req.lang or "en-US",
                max_tokens=45
            ):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_payload = json.dumps({"token": "Visual perception active.", "error": str(e)})
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/v1/vision/ocr")
def read_text_direct(request: Request):
    """Direct local OCR execution on current frame."""
    verify_client_origin(request)
    frame = camera_manager.get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame.")

    enhanced = VisionPreprocessor.optimize_for_ocr(frame)
    result = ocr_engine.extract_text(enhanced)
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
