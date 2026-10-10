import cv2
import time
from vision.camera import CameraManager
from vision.preprocessor import VisionPreprocessor
from vision.ocr import LocalOCREngine
from vision.router import VisionRouter

def run_ocr_test():
    print("==================================================")
    print(" EchoVision - Phase 1 Local OCR & Triage Test")
    print("==================================================")
    print("Hold up any printed text, book, medicine bottle, or sign.")
    print("Commands:")
    print("  'space' -> Capture & Run Local OCR Triage")
    print("  'c'     -> Switch camera index (0 / 1)")
    print("  'q'     -> Quit")
    print("==================================================")

    cam_index = 0
    cam = CameraManager(cam_index)
    if not cam.start():
        print(f"Could not open camera {cam_index}, trying camera 1...")
        cam_index = 1
        cam = CameraManager(cam_index)
        if not cam.start():
            print("No camera found. Exiting.")
            return

    ocr_engine = LocalOCREngine()
    router = VisionRouter(ocr_engine)

    if not ocr_engine.is_available:
        print("\n[NOTE] Tesseract-OCR is not currently installed or in PATH on your Windows system.")
        print("Install it via: winget install UB-Mannheim.TesseractOCR (or download installer).")
        print("Camera preview and preprocessing are still active!\n")

    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        enhanced = VisionPreprocessor.optimize_for_ocr(frame)

        # Show feeds
        cv2.imshow("EchoVision - Raw Hardware Feed (Press SPACE to Read)", frame)
        cv2.imshow("EchoVision - Optimized AI Feed", enhanced)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cam.release()
            cam_index = 1 if cam_index == 0 else 0
            cam = CameraManager(cam_index)
            cam.start()
            print(f"Switched to camera {cam_index}")
        elif key == 32:  # Spacebar pressed
            print("\n🔍 Analyzing frame...")
            
            # 1. OCR directly on the optimized FULL frame (no aggressive cropping risk)
            decision = router.evaluate_frame(enhanced, user_intent="read this label")

            print("-" * 50)
            print(f"Decision Route : {decision['route']} (Tier {decision['tier']})")
            print(f"Cost Incurred  : ${0.00 if not decision['cost_incurred'] else '0.002 (Cloud)'}")
            print(f"Reason         : {decision['reason']}")
            
            ocr_info = decision['ocr_data']
            if ocr_info['success']:
                print(f"Extracted Words: {ocr_info['word_count']}")
                print(f"Confidence     : {ocr_info['confidence']}%")
                print(f"Extracted Text :\n\"{ocr_info['text']}\"")
            else:
                print(f"OCR Status     : {ocr_info.get('error', 'No text extracted')}")
            print("-" * 50 + "\n")

    cam.release()
    cv2.destroyAllWindows()
    print("Test finished.")

if __name__ == "__main__":
    run_ocr_test()
