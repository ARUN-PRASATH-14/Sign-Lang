"""
Gesture Dataset Collector Utility.
Captures hand gesture images and coordinates directly from webcam feed.
Creates structured folders inside dataset/<gesture_label>/
"""

import os
import time
import cv2
from utils.hand_detector import HandDetector


def collect_gesture_data(
    gesture_name: str,
    target_count: int = 50,
    camera_index: int = 0,
    save_folder: str = "dataset",
):
    target_dir = os.path.join(save_folder, gesture_name)
    os.makedirs(target_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(target_dir) if f.endswith(".jpg")]
    start_index = len(existing_files)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] Error: Could not access camera {camera_index}")
        return

    detector = HandDetector(max_hands=1, detection_con=0.7)

    print(f"\n[*] Preparing to capture samples for gesture: '{gesture_name}'")
    print(f"[*] Target: {target_count} images in {target_dir}")
    print("[*] Controls:")
    print("    - Press 'SPACE' to start/pause automatic capturing")
    print("    - Press 'c' to capture a single sample")
    print("    - Press 'q' to quit\n")

    capturing_auto = False
    count = 0
    last_capture_time = 0

    while count < target_count:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        display_frame, results = detector.find_hands(display_frame, draw=True)
        lm_list, bbox, handedness = detector.get_hand_landmarks(display_frame)

        # Status text
        status_str = "AUTO CAPTURING..." if capturing_auto else "READY (Press SPACE or 'c')"
        color = (0, 255, 0) if capturing_auto else (0, 255, 255)

        cv2.putText(
            display_frame,
            f"Gesture: {gesture_name} | Progress: {count}/{target_count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"Status: {status_str}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        now = time.time()
        should_save = False

        if capturing_auto and (now - last_capture_time > 0.35) and bbox is not None:
            should_save = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            capturing_auto = not capturing_auto
        elif key == ord("c") and bbox is not None:
            should_save = True
        elif key == ord("q"):
            break

        if should_save:
            count += 1
            filename = os.path.join(target_dir, f"img_{start_index + count:04d}.jpg")
            cv2.imwrite(filename, frame)
            last_capture_time = now
            # Visual flash effect
            cv2.circle(display_frame, (30, 100), 12, (0, 255, 0), -1)

        cv2.imshow("Dataset Collector", display_frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[+] Completed capturing {count} samples for '{gesture_name}' in '{target_dir}'.")


if __name__ == "__main__":
    gesture = input("Enter gesture name to collect (e.g. Hello, Thank You, A): ").strip()
    if gesture:
        collect_gesture_data(gesture, target_count=40)
