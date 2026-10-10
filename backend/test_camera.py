import cv2
import time
from vision.camera import CameraManager
from vision.preprocessor import VisionPreprocessor

def run_hardware_test():
    """
    Standalone visual test for the local camera and OpenCV preprocessor.
    Allows the user to see exactly what the AI will see in real-time.
    """
    print("==================================================")
    print(" EchoVision - Hardware Diagnostics & Core Test")
    print("==================================================")
    print("Press 'q' on the video window to exit.")
    print("Press 'c' to change camera index (0 to 1, etc).")
    
    cam_index = 0
    cam = CameraManager(cam_index)
    
    if not cam.start():
        print("Attempting fallback to camera index 1...")
        cam_index = 1
        cam = CameraManager(cam_index)
        if not cam.start():
            print("Failed to access any local cameras.")
            return

    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Run our custom preprocessing pipeline
        enhanced = VisionPreprocessor.optimize_for_ocr(frame)
        
        # Display the results
        cv2.imshow("EchoVision - Raw Hardware Feed", frame)
        cv2.imshow("EchoVision - Optimized AI Feed (What the Brain sees)", enhanced)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Switch camera (e.g., from laptop to external phone cam)
            cam.release()
            cam_index = 1 if cam_index == 0 else 0
            cam = CameraManager(cam_index)
            cam.start()

    cam.release()
    cv2.destroyAllWindows()
    print("Diagnostics complete.")

if __name__ == "__main__":
    run_hardware_test()
