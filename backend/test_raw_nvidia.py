import cv2
import base64
from vision.cloud_brain import CloudMultimodalEngine

brain = CloudMultimodalEngine()

# Read the RAW color image
raw_frame = cv2.imread("debug_raw.jpg")

# Downscale to 720p or keep high quality color JPEG
_, buffer = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64 = base64.b64encode(buffer).decode('utf-8')

print("Sending RAW COLOR image to NVIDIA...")
result = brain.analyze_image(b64, "Describe the person or scene in front of me.")
print("NVIDIA RESULT:")
print(result)
