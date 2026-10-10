import cv2
from vision.camera import CameraManager
from vision.preprocessor import VisionPreprocessor

cam = CameraManager(0)
cam.start()

# Flush a couple frames because webcam auto-exposure needs 2-3 frames to settle
for _ in range(5):
    cam.get_frame()

frame = cam.get_frame()
cam.release()

if frame is not None:
    cv2.imwrite("debug_raw.jpg", frame)
    enhanced = VisionPreprocessor.optimize_for_ocr(frame)
    cv2.imwrite("debug_enhanced.jpg", enhanced)
    print(f"Captured frame shape: {frame.shape}")
    print(f"Mean brightness (raw): {frame.mean()}")
    print("Saved debug_raw.jpg and debug_enhanced.jpg")
else:
    print("Could not capture frame.")
