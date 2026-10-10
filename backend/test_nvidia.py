import cv2
import base64
from vision.cloud_brain import CloudMultimodalEngine

print("Initializing NVIDIA Cloud Brain...")
brain = CloudMultimodalEngine()

# Grab a live frame from webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("Could not grab camera frame for test.")
    exit(1)

# Compress to JPEG
_, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
b64 = base64.b64encode(buffer).decode('utf-8')

print("Sending frame to NVIDIA NIM (meta/llama-3.2-11b-vision-instruct)...")
result = brain.analyze_image(b64, "Describe what is in front of me and read any visible text.")
print("-" * 50)
print("NVIDIA Response:")
print(result)
print("-" * 50)
