"""
Real-Time Sign Language Recognition & Translation Script.
Replicates the visual interface shown in the presentation:
- MediaPipe Hand Skeleton Tracking
- Green bounding box with handedness & recognized gesture: e.g. "Right: O"
- Audio Text-to-Speech (TTS) integration
- Live sentence builder
"""

import json
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.architecture import SignLandmarkMLP, SignCNNLSTM
from utils.hand_detector import HandDetector
from utils.tts_engine import speak_text
from utils.gesture_recognizer import BiomechanicalGestureRecognizer


def run_realtime_prediction(
    model_type: str = "mlp",
    camera_index: int = 0,
    confidence_thresh: float = 0.65,
):
    print("=" * 60)
    print("  REAL-TIME SIGN LANGUAGE RECOGNITION (OpenCV + MediaPipe)")
    print("=" * 60)

    # 1. Load Label Encoder
    encoder_path = os.path.join("models", "label_encoder.json")
    if not os.path.exists(encoder_path):
        print("[!] Model labels not found. Running train_initial_model.py...")
        from train_initial_model import train_models

        train_models()

    with open(encoder_path, "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"[*] Loaded {num_classes} gesture classes.")

    # 2. Load PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "cnn_lstm" and os.path.exists(
        os.path.join("models", "sign_model_cnn_lstm.pth")
    ):
        model = SignCNNLSTM(input_dim=63, num_classes=num_classes, hidden_dim=64)
        model.load_state_dict(
            torch.load(os.path.join("models", "sign_model_cnn_lstm.pth"), map_location=device)
        )
        print("[*] Loaded Hybrid CNN + LSTM Model.")
    else:
        model = SignLandmarkMLP(input_dim=63, num_classes=num_classes)
        model.load_state_dict(
            torch.load(os.path.join("models", "sign_model.pth"), map_location=device)
        )
        print("[*] Loaded SignLandmarkMLP Model.")

    model.to(device)
    model.eval()

    # 3. Setup Webcam and Hand Detector
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {camera_index}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(mode=False, max_hands=2, detection_con=0.7, track_con=0.6)

    # State tracking
    current_sentence = []
    last_spoken_sign = ""
    last_detected_sign = ""
    stable_count = 0
    stable_threshold = 12  # frames of consistent gesture before confirming

    fps_time = time.time()
    fps_counter = 0
    display_fps = 0

    print("\n[*] CONTROLS:")
    print("    - Press 's': Speak current gesture via Text-to-Speech (TTS)")
    print("    - Press 'a': Add current gesture to sentence")
    print("    - Press 'space': Speak entire sentence")
    print("    - Press 'c': Clear current sentence")
    print("    - Press 'q': Exit application\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            display_fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        # Mirror frame
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Find hands
        frame, results = detector.find_hands(frame, draw=True)

        recognized_this_frame = None
        recognized_conf = 0.0

        if results and results.multi_hand_landmarks:
            for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                lm_list, bbox, handedness = detector.get_hand_landmarks(frame, hand_idx)
                features = detector.extract_normalized_features(hand_lms)

                if features is not None and bbox is not None:
                    raw_lms = detector.get_raw_landmarks(hand_idx)
                    recognizer = BiomechanicalGestureRecognizer()
                    geo_res = recognizer.analyze_hand(raw_lms) if raw_lms is not None else None

                    # Model Inference
                    tensor_feat = (
                        torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                    )
                    with torch.no_grad():
                        logits = model(tensor_feat)
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_idx = int(np.argmax(probs))
                        ml_conf = float(probs[pred_idx])
                        ml_sign = label_map.get(str(pred_idx), "Unknown")

                    # Ensemble: prioritize biometric landmark geometry
                    if geo_res and geo_res["confidence"] >= 0.80:
                        pred_sign = geo_res["label"]
                        conf = geo_res["confidence"]
                    elif ml_conf >= confidence_thresh and ml_sign != "Unknown":
                        pred_sign = ml_sign
                        conf = ml_conf
                    elif geo_res:
                        pred_sign = geo_res["label"]
                        conf = geo_res["confidence"]
                    else:
                        pred_sign = "Unknown"
                        conf = 0.0

                    if conf >= 0.60 and pred_sign != "Unknown":
                        recognized_this_frame = pred_sign
                        recognized_conf = conf

                        x1, y1, x2, y2 = bbox
                        # Padding for bounding box
                        pad = 25
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)

                        # Draw green bounding box (as seen in sample output)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label above bounding box: "Right: O" (green/blue style)
                        label_text = f"{handedness}: {pred_sign}"
                        conf_text = f"{conf * 100:.1f}%"

                        # Drop shadow / double text as in Slide 5
                        cv2.putText(
                            frame,
                            label_text,
                            (x1, max(30, y1 - 35)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 0, 0),  # Blue shadow
                            3,
                        )
                        cv2.putText(
                            frame,
                            label_text,
                            (x1, max(30, y1 - 35)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),  # Green text
                            2,
                        )
                        cv2.putText(
                            frame,
                            conf_text,
                            (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

        # Stabilization logic
        if recognized_this_frame is not None:
            if recognized_this_frame == last_detected_sign:
                stable_count += 1
            else:
                last_detected_sign = recognized_this_frame
                stable_count = 1

            if stable_count == stable_threshold:
                # Add to sentence automatically
                if not current_sentence or current_sentence[-1] != recognized_this_frame:
                    current_sentence.append(recognized_this_frame)
                    # Automatically speak if it's a greeting, word, or common term
                    common_terms_speak = [
                        "Hello", "Thank You", "Please", "Sorry", "Yes", "No",
                        "I Love You", "Peace", "Help / Good", "Bad", "Water",
                        "Food / Eat", "Okay", "Stop", "Call Me", "Rock On", "You"
                    ]
                    if recognized_this_frame in common_terms_speak:
                        # Clean label for natural TTS speech
                        spoken_phrase = recognized_this_frame.replace(" / ", " or ")
                        speak_text(spoken_phrase)
        else:
            stable_count = 0

        # Bottom HUD Banner for Sentence & Status
        hud_h = 75
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - hud_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        sentence_str = " ".join(current_sentence) if current_sentence else "[No signs recorded]"
        cv2.putText(
            frame,
            f"Sentence: {sentence_str}",
            (20, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Top diagnostic bar
        cv2.putText(
            frame,
            f"FPS: {display_fps:.1f} | Signs: {num_classes} | [s]=Speak, [c]=Clear, [q]=Quit",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and recognized_this_frame:
            speak_text(recognized_this_frame)
        elif key == ord("a") and recognized_this_frame:
            current_sentence.append(recognized_this_frame)
        elif key == ord(" ") and current_sentence:
            speak_text(" ".join(current_sentence))
        elif key == ord("c"):
            current_sentence.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_prediction()
