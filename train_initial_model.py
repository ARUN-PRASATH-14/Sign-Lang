"""
Generates initial baseline dataset with biomechanical landmark geometry,
performs augmentation, and trains the CNN-LSTM and MLP models.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from models.architecture import SignCNNLSTM, SignLandmarkMLP


# 21 Hand Landmark indices reference:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

CLASSES = [
    "A", "B", "C", "D", "E", "F", "I", "L", "O",
    "V", "W", "Y", "Hello", "Thank You", "Yes", "No", "I Love You", "Peace"
]


def get_base_hand_template():
    """Returns typical open-hand 21 3D landmarks relative to wrist (0, 0, 0)."""
    return np.array([
        [0.0, 0.0, 0.0],       # 0: Wrist
        [-0.15, -0.15, -0.02], # 1: Thumb CMC
        [-0.25, -0.28, -0.04], # 2: Thumb MCP
        [-0.32, -0.42, -0.06], # 3: Thumb IP
        [-0.38, -0.55, -0.08], # 4: Thumb TIP
        [-0.15, -0.45, -0.02], # 5: Index MCP
        [-0.18, -0.65, -0.03], # 6: Index PIP
        [-0.20, -0.80, -0.04], # 7: Index DIP
        [-0.22, -0.95, -0.05], # 8: Index TIP
        [0.00, -0.47, 0.00],   # 9: Middle MCP
        [0.00, -0.68, -0.01],  # 10: Middle PIP
        [0.00, -0.85, -0.02],  # 11: Middle DIP
        [0.00, -1.00, -0.03],  # 12: Middle TIP
        [0.15, -0.44, -0.01],  # 13: Ring MCP
        [0.17, -0.64, -0.02],  # 14: Ring PIP
        [0.19, -0.79, -0.03],  # 15: Ring DIP
        [0.20, -0.92, -0.04],  # 16: Ring TIP
        [0.28, -0.38, -0.02],  # 17: Pinky MCP
        [0.32, -0.54, -0.03],  # 18: Pinky PIP
        [0.35, -0.67, -0.04],  # 19: Pinky DIP
        [0.38, -0.79, -0.05],  # 20: Pinky TIP
    ], dtype=np.float32)


def curl_finger(coords, mcp_idx, tip_idx, curl_factor=0.7):
    """Curls a finger inward toward palm."""
    coords = coords.copy()
    mcp = coords[mcp_idx]
    for i in range(mcp_idx + 1, tip_idx + 1):
        progress = (i - mcp_idx) / (tip_idx - mcp_idx)
        coords[i, 1] = mcp[1] + (coords[i, 1] - mcp[1]) * (1 - curl_factor)
        coords[i, 2] += 0.2 * progress * curl_factor
        coords[i, 0] = mcp[0] * (1.0 - 0.2 * progress)
    return coords


def create_gesture_template(gesture_name: str):
    """Synthesizes biomechanical template for each sign."""
    coords = get_base_hand_template()

    if gesture_name == "A":
        # Fist with thumb upright against index side
        coords = curl_finger(coords, 5, 8, 0.85)
        coords = curl_finger(coords, 9, 12, 0.85)
        coords = curl_finger(coords, 13, 16, 0.85)
        coords = curl_finger(coords, 17, 20, 0.85)
        # Thumb upright beside index
        coords[4] = [-0.18, -0.55, -0.05]

    elif gesture_name == "B":
        # 4 fingers straight together, thumb folded over palm
        coords[8] = [-0.08, -0.96, -0.02]
        coords[12] = [0.00, -1.00, -0.02]
        coords[16] = [0.08, -0.95, -0.02]
        coords[20] = [0.15, -0.88, -0.02]
        # Thumb across palm
        coords[4] = [0.05, -0.35, 0.05]

    elif gesture_name == "C":
        # Hand curved in C-shape
        coords = curl_finger(coords, 5, 8, 0.45)
        coords = curl_finger(coords, 9, 12, 0.45)
        coords = curl_finger(coords, 13, 16, 0.45)
        coords = curl_finger(coords, 17, 20, 0.45)
        coords[4] = [-0.25, -0.40, 0.15]

    elif gesture_name == "D":
        # Index straight up, thumb touching middle, ring, pinky tips
        coords = curl_finger(coords, 9, 12, 0.75)
        coords = curl_finger(coords, 13, 16, 0.75)
        coords = curl_finger(coords, 17, 20, 0.75)
        coords[4] = [0.00, -0.45, 0.12]
        coords[8] = [-0.05, -0.98, -0.02]

    elif gesture_name == "E":
        # All fingers tightly curled with thumb tucked under
        coords = curl_finger(coords, 5, 8, 0.88)
        coords = curl_finger(coords, 9, 12, 0.88)
        coords = curl_finger(coords, 13, 16, 0.88)
        coords = curl_finger(coords, 17, 20, 0.88)
        coords[4] = [0.00, -0.30, 0.10]

    elif gesture_name == "F":
        # Index and thumb touching circle, middle/ring/pinky spread up
        coords[4] = [-0.10, -0.55, 0.08]
        coords[8] = [-0.10, -0.55, 0.08]
        coords[12] = [0.00, -1.00, -0.02]
        coords[16] = [0.15, -0.95, -0.02]
        coords[20] = [0.28, -0.85, -0.02]

    elif gesture_name == "I":
        # Pinky extended, all other fingers curled in fist
        coords = curl_finger(coords, 5, 8, 0.85)
        coords = curl_finger(coords, 9, 12, 0.85)
        coords = curl_finger(coords, 13, 16, 0.85)
        coords[4] = [-0.10, -0.38, 0.08]
        coords[20] = [0.25, -0.85, -0.02]

    elif gesture_name == "L":
        # Thumb and index form 90 deg L shape, others curled
        coords = curl_finger(coords, 9, 12, 0.85)
        coords = curl_finger(coords, 13, 16, 0.85)
        coords = curl_finger(coords, 17, 20, 0.85)
        coords[4] = [-0.48, -0.25, -0.02]  # Extended thumb horizontally
        coords[8] = [-0.10, -0.98, -0.02]  # Extended index vertically

    elif gesture_name == "O":
        # All fingertips touch thumb tip forming O shape (Sample Output screenshot)
        o_touch_point = [-0.10, -0.55, 0.12]
        coords[4] = o_touch_point
        coords = curl_finger(coords, 5, 8, 0.65)
        coords = curl_finger(coords, 9, 12, 0.65)
        coords = curl_finger(coords, 13, 16, 0.65)
        coords = curl_finger(coords, 17, 20, 0.65)
        coords[8] = [-0.08, -0.56, 0.10]
        coords[12] = [-0.05, -0.56, 0.10]

    elif gesture_name == "V" or gesture_name == "Peace":
        # Index and middle spread in V, others curled
        coords = curl_finger(coords, 13, 16, 0.85)
        coords = curl_finger(coords, 17, 20, 0.85)
        coords[4] = [0.00, -0.35, 0.08]
        coords[8] = [-0.18, -0.95, -0.02]
        coords[12] = [0.12, -0.96, -0.02]

    elif gesture_name == "W":
        # Index, middle, ring spread upright, pinky and thumb curled
        coords = curl_finger(coords, 17, 20, 0.85)
        coords[4] = [0.08, -0.35, 0.08]
        coords[8] = [-0.20, -0.95, -0.02]
        coords[12] = [0.00, -1.00, -0.02]
        coords[16] = [0.20, -0.95, -0.02]

    elif gesture_name == "Y":
        # Thumb and pinky extended, middle 3 curled
        coords = curl_finger(coords, 5, 8, 0.85)
        coords = curl_finger(coords, 9, 12, 0.85)
        coords = curl_finger(coords, 13, 16, 0.85)
        coords[4] = [-0.45, -0.25, -0.04]
        coords[20] = [0.42, -0.65, -0.04]

    elif gesture_name == "Hello":
        # Open 5 palm slightly tilted forward / wave
        coords[4] = [-0.35, -0.45, -0.02]
        coords[8] = [-0.15, -0.96, 0.05]
        coords[12] = [0.00, -1.00, 0.05]
        coords[16] = [0.15, -0.94, 0.05]
        coords[20] = [0.30, -0.85, 0.05]

    elif gesture_name == "Thank You":
        # Flat hand palm facing chin/inward
        coords = curl_finger(coords, 5, 8, 0.15)
        coords = curl_finger(coords, 9, 12, 0.15)
        coords = curl_finger(coords, 13, 16, 0.15)
        coords = curl_finger(coords, 17, 20, 0.15)
        coords[4] = [-0.20, -0.45, 0.15]

    elif gesture_name == "Yes":
        # S-fist closed
        coords = curl_finger(coords, 5, 8, 0.90)
        coords = curl_finger(coords, 9, 12, 0.90)
        coords = curl_finger(coords, 13, 16, 0.90)
        coords = curl_finger(coords, 17, 20, 0.90)
        coords[4] = [0.02, -0.48, 0.12]  # Thumb across folded fingers

    elif gesture_name == "No":
        # Index and middle snapped onto thumb
        coords = curl_finger(coords, 13, 16, 0.85)
        coords = curl_finger(coords, 17, 20, 0.85)
        coords[4] = [-0.08, -0.50, 0.08]
        coords[8] = [-0.08, -0.52, 0.08]
        coords[12] = [-0.04, -0.52, 0.08]

    elif gesture_name == "I Love You":
        # Thumb, Index, and Pinky extended, Middle and Ring curled
        coords = curl_finger(coords, 9, 12, 0.85)
        coords = curl_finger(coords, 13, 16, 0.85)
        coords[4] = [-0.42, -0.28, -0.04]
        coords[8] = [-0.15, -0.98, -0.02]
        coords[20] = [0.38, -0.75, -0.04]

    # Normalize template relative to wrist (0) and max distance
    wrist = coords[0]
    rel = coords - wrist
    max_d = np.max(np.linalg.norm(rel, axis=1))
    if max_d > 1e-6:
        rel = rel / max_d
    return rel


def augment_landmarks(template, count=250):
    """
    Applies 3D rotation, scaling, and Gaussian noise to generate realistic variations.
    """
    samples = []
    for _ in range(count):
        # 3D random rotation angle
        angle_z = np.random.uniform(-0.15, 0.15)
        angle_y = np.random.uniform(-0.10, 0.10)
        angle_x = np.random.uniform(-0.10, 0.10)

        # Rotation matrix Z
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        # Rotation matrix Y
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

        # Rotation matrix X
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

        R = Rz @ Ry @ Rx
        rotated = template @ R.T

        # Scale jitter
        scale = np.random.uniform(0.92, 1.08)
        scaled = rotated * scale

        # Micro Gaussian jitter on joints (more on tips)
        noise = np.random.normal(0, 0.012, size=scaled.shape)
        # Add higher jitter on fingertip indices [4, 8, 12, 16, 20]
        for tip in [4, 8, 12, 16, 20]:
            noise[tip] += np.random.normal(0, 0.02, size=3)

        sample = scaled + noise

        # Re-normalize to wrist
        sample = sample - sample[0]
        max_d = np.max(np.linalg.norm(sample, axis=1))
        if max_d > 1e-6:
            sample = sample / max_d

        samples.append(sample.flatten().astype(np.float32))

    return np.array(samples)


def train_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    print("=== Generating Sign Language Biomechanical Dataset ===")
    all_features = []
    all_labels = []

    for idx, cls_name in enumerate(CLASSES):
        template = create_gesture_template(cls_name)
        augmented = augment_landmarks(template, count=250)
        all_features.append(augmented)
        all_labels.extend([idx] * len(augmented))
        print(f"  [+] Generated {len(augmented)} samples for '{cls_name}'")

    X = np.vstack(all_features)
    y = np.array(all_labels, dtype=np.int64)

    # Save processed numpy arrays
    np.save("processed/landmarks_data.npy", X)
    np.save("processed/labels.npy", y)
    with open("models/label_encoder.json", "w") as f:
        json.dump({str(i): name for i, name in enumerate(CLASSES)}, f, indent=2)

    print(f"\nTotal Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(CLASSES)} classes.")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # 1. Train MLP Model (SignLandmarkMLP)
    print("\n--- Training SignLandmarkMLP ---")
    mlp_model = SignLandmarkMLP(input_dim=63, num_classes=len(CLASSES))
    optimizer_mlp = torch.optim.AdamW(mlp_model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 41):
        mlp_model.train()
        optimizer_mlp.zero_grad()
        out = mlp_model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer_mlp.step()

        if epoch % 10 == 0:
            mlp_model.eval()
            with torch.no_grad():
                val_out = mlp_model(X_val_t)
                val_loss = criterion(val_out, y_val_t)
                val_acc = (val_out.argmax(dim=1) == y_val_t).float().mean().item() * 100
                print(f"  Epoch {epoch:02d}/40 - Loss: {loss.item():.4f} - Val Acc: {val_acc:.2f}%")

    torch.save(mlp_model.state_dict(), "models/sign_model.pth")
    print("Saved models/sign_model.pth successfully!")

    # 2. Train Hybrid CNN-LSTM Model (SignCNNLSTM)
    print("\n--- Training Hybrid CNN + LSTM Model ---")
    cnn_lstm_model = SignCNNLSTM(input_dim=63, num_classes=len(CLASSES), hidden_dim=64)
    optimizer_cl = torch.optim.AdamW(cnn_lstm_model.parameters(), lr=0.002, weight_decay=1e-4)

    for epoch in range(1, 41):
        cnn_lstm_model.train()
        optimizer_cl.zero_grad()
        out = cnn_lstm_model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer_cl.step()

        if epoch % 10 == 0:
            cnn_lstm_model.eval()
            with torch.no_grad():
                val_out = cnn_lstm_model(X_val_t)
                val_acc = (val_out.argmax(dim=1) == y_val_t).float().mean().item() * 100
                print(f"  Epoch {epoch:02d}/40 - Val Acc: {val_acc:.2f}%")

    torch.save(cnn_lstm_model.state_dict(), "models/sign_model_cnn_lstm.pth")
    print("Saved models/sign_model_cnn_lstm.pth successfully!")
    print("\n[ALL MODELS READY FOR INFERENCE AND STREAMLIT!]")


if __name__ == "__main__":
    train_models()
