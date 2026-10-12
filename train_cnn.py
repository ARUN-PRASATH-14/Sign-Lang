"""
Deep Learning Model Training Script for Sign Language Recognition.
Trains CNN, LSTM, and Hybrid Deep Landmark models.
Outputs accuracy metrics, loss progression, and saves trained weights.
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from models.architecture import SignCNNLSTM, SignLandmarkMLP


def train_pipeline(
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    model_type: str = "mlp",  # "mlp" or "cnn_lstm"
):
    print("=" * 60)
    print("  SIGN LANGUAGE DEEP LEARNING MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Preprocessed Data
    data_path = os.path.join("processed", "landmarks_data.npy")
    label_path = os.path.join("processed", "labels.npy")
    encoder_path = os.path.join("models", "label_encoder.json")

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        print("[!] Preprocessed dataset not found. Generating baseline...")
        from train_initial_model import train_models

        train_models()
        return

    X = np.load(data_path)
    y = np.load(label_path)

    with open(encoder_path, "r") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    num_classes = len(class_names)

    print(f"[*] Dataset Loaded: {X.shape[0]} total samples | {X.shape[1]} features")
    print(f"[*] Total Classes ({num_classes}): {class_names}\n")

    # 2. Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # 3. Model Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    if model_type == "cnn_lstm":
        model = SignCNNLSTM(input_dim=63, num_classes=num_classes, hidden_dim=64).to(device)
        save_path = os.path.join("models", "sign_model_cnn_lstm.pth")
    else:
        model = SignLandmarkMLP(input_dim=63, num_classes=num_classes).to(device)
        save_path = os.path.join("models", "sign_model.pth")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # 4. Training Loop
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_acc = 0.0
    start_time = time.time()

    dataset_size = len(X_train)
    indices = np.arange(dataset_size)

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        batches = 0

        for b_start in range(0, dataset_size, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            b_x = X_train_t[b_idx].to(device)
            b_y = y_train_t[b_idx].to(device)

            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_train_loss = epoch_loss / max(batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device))
            val_loss = criterion(val_preds, y_val_t.to(device)).item()
            val_pred_labels = val_preds.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_pred_labels) * 100

        scheduler.step(val_acc)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs}] "
                f"| Train Loss: {avg_train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f} "
                f"| Val Acc: {val_acc:.2f}%"
            )

    elapsed = time.time() - start_time
    print(f"\n[+] Training finished in {elapsed:.1f}s. Best Val Accuracy: {best_acc:.2f}%")
    print(f"[+] Model checkpoint saved to: {save_path}")

    # Save history metrics
    history_file = os.path.join("models", "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    # Classification Report
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        final_preds = model(X_val_t.to(device)).argmax(dim=1).cpu().numpy()

    print("\n--- Final Validation Classification Report ---")
    print(
        classification_report(
            y_val, final_preds, target_names=class_names, zero_division=0
        )
    )
    return history


if __name__ == "__main__":
    train_pipeline(epochs=40, model_type="mlp")
