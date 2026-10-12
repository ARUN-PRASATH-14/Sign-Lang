"""
Dataset Preprocessing Pipeline.
Iterates through all gesture image folders in dataset/,
detects hand landmarks using MediaPipe, extracts normalized 63-D feature vectors,
and compiles them into NumPy arrays ready for model training.
"""

import json
import os
import cv2
import numpy as np
from utils.hand_detector import HandDetector


def preprocess_all_datasets(
    dataset_dir: str = "dataset",
    output_dir: str = "processed",
    models_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(f"[*] Dataset directory '{dataset_dir}' is empty.")
        print("[*] Generating baseline biometric dataset using train_initial_model.py...")
        from train_initial_model import train_models

        train_models()
        return

    # Scan gesture folders
    classes = sorted(
        [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    )
    if not classes:
        print("[!] No class folders found in dataset/.")
        return

    print(f"[*] Found {len(classes)} classes: {classes}")
    label_map = {str(i): name for i, name in enumerate(classes)}

    detector = HandDetector(mode=True, max_hands=1, detection_con=0.5)

    all_features = []
    all_labels = []

    for label_idx, cls_name in enumerate(classes):
        cls_folder = os.path.join(dataset_dir, cls_name)
        img_files = [
            f
            for f in os.listdir(cls_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"[*] Processing '{cls_name}' ({len(img_files)} images)...")

        valid_count = 0
        for img_name in img_files:
            img_path = os.path.join(cls_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            _, results = detector.find_hands(img, draw=False)
            features = detector.get_features_for_model(hand_index=0)

            if features is not None and len(features) == 63:
                all_features.append(features)
                all_labels.append(label_idx)
                valid_count += 1

        print(f"    -> Extracted {valid_count} valid landmark sets.")

    if not all_features:
        print("[!] Could not extract any valid hand landmarks from images.")
        return

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Save to processed/
    np.save(os.path.join(output_dir, "landmarks_data.npy"), X)
    np.save(os.path.join(output_dir, "labels.npy"), y)

    # Save label encoder JSON
    with open(os.path.join(models_dir, "label_encoder.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n[+] Successfully saved {len(X)} samples to {output_dir}/landmarks_data.npy")
    print(f"[+] Classes saved to {models_dir}/label_encoder.json")


if __name__ == "__main__":
    preprocess_all_datasets()
