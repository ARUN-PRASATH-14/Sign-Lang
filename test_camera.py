"""
Camera Diagnostic & Test Utility.
Verifies camera connectivity, frame capture, resolution, and real-time FPS.
"""

import time
import cv2


def test_webcam(camera_index: int = 0):
    print(f"[*] Attempting to open webcam at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[!] Error: Could not open camera at index {camera_index}.")
        print("[!] Tip: Verify camera permissions in Windows Settings -> Privacy -> Camera.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[+] Camera opened successfully!")
    print(f"    - Resolution: {width} x {height}")
    print(f"    - Reported FPS: {fps}")
    print("\n[*] Press 'q' in the window to exit camera test.\n")

    prev_time = time.time()
    frame_count = 0
    actual_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to read frame from camera.")
            break

        frame_count += 1
        curr_time = time.time()
        if curr_time - prev_time >= 1.0:
            actual_fps = frame_count / (curr_time - prev_time)
            frame_count = 0
            prev_time = curr_time

        # Flip horizontally for selfie mirror effect
        frame = cv2.flip(frame, 1)

        # Draw diagnostic info overlay
        cv2.putText(
            frame,
            f"Camera Test | Resolution: {width}x{height} | FPS: {actual_fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' to Close Test",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Camera Diagnostic Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Camera test completed.")
    return True


if __name__ == "__main__":
    test_webcam(0)
