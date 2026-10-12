# SMART COMMUNICATION SYSTEM FOR DEAF AND DUMB USING AI-BASED SIGN LANGUAGE TRANSLATION

**Institution:** K.L.N. College of Engineering (An Autonomous Institution, Affiliated to Anna University, Chennai)  
**Department:** Computer Science and Engineering (Internet of Things)  
**Guide:** Mrs. Roselin Reeta P, Assistant Professor 2  
**Team Members:** Joshua J [910623110021], Hazi Khan U [910623110016]  
**Domain:** Deep Learning (DL) & Machine Learning (ML)

---

## 🌟 Overview

This project provides a **software-based two-way AI communication bridge** between deaf/mute individuals and hearing people using standard consumer webcams and microphones with **zero specialized hardware required**.

### Two-Way Communication Architecture:
1. **Forward Flow (Deaf/Mute ➡️ Hearing Individual):**
   - Live Webcam Feed captures hand gestures.
   - **Google MediaPipe** extracts 21 3D hand landmarks per hand (63 features).
   - Biomechanical normalization provides distance- and translation-invariant coordinates.
   - Deep Learning Model (**1D-CNN + Bidirectional LSTM / Deep MLP**) classifies signs in real time.
   - Sentence builder accumulates words into coherent phrases.
   - **Text-to-Speech (TTS Engine)** speaks the message aloud through the speakers.
2. **Reverse Flow (Hearing Person ➡️ Deaf/Mute Individual):**
   - Normal individual speaks into the microphone.
   - **Speech-to-Text (STT Engine)** transcribes the speech into text.
   - Displayed in large, high-contrast visual captions and dynamic finger-spelling cues on screen.

---

## 📁 Project Structure

```
sasi/
│
├── dataset/                  # Collected raw gesture images (folders per class)
├── models/                   # Deep learning architectures and saved weights
│   ├── architecture.py       # CNN + LSTM & Landmark MLP PyTorch models
│   ├── label_encoder.json    # JSON mapping of class IDs to gesture labels
│   ├── sign_model.pth        # Trained primary deep model weights
│   └── sign_model_cnn_lstm.pth # Trained hybrid CNN-LSTM weights
├── processed/                # Normalized preprocessed landmark datasets
│   ├── landmarks_data.npy    # 63-dimensional normalized features
│   └── labels.npy            # Numerical category labels
├── utils/                    # Core pipeline utilities
│   ├── __init__.py
│   ├── hand_detector.py      # MediaPipe hand tracker & feature normalizer
│   ├── tts_engine.py         # Text-to-Speech audio synthesizer (pyttsx3)
│   └── stt_engine.py         # Speech-to-Text microphone recognizer
│
├── app.py                    # Complete Streamlit Interactive Web Application
├── predict.py                # Standalone OpenCV Real-Time Window (Slide 5 replica)
├── train_cnn.py              # Model training script with validation & metrics
├── preprocess_dataset.py     # Extracts MediaPipe landmarks from dataset/
├── dataset_collector.py      # Interactive webcam gesture recorder
├── test_camera.py            # Camera diagnostic tool
├── train_initial_model.py    # Generates initial biomechanical baseline models
├── requirements.txt          # Python dependencies
├── run_app.bat               # Windows double-click application launcher
└── README.md                 # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Launch the Streamlit Web Application (Recommended)
You can double-click `run_app.bat` or run:
```bash
py -3.12 -m streamlit run app.py
```
This opens the web interface in your browser at `http://localhost:8501`.

### 2. Run the Standalone OpenCV Inference Window (as in Slide 5)
```bash
py -3.12 predict.py
```
**Controls in OpenCV window:**
- `s`: Speak current gesture via Text-to-Speech (TTS)
- `a`: Add detected gesture to accumulated sentence
- `Space`: Speak entire sentence
- `c`: Clear sentence
- `q`: Exit

### 3. Record New Custom Signs
```bash
py -3.12 dataset_collector.py
```

### 4. Train/Retrain the Neural Network Model
```bash
py -3.12 train_cnn.py
```

### 5. Camera Health Diagnostic
```bash
py -3.12 test_camera.py
```

---

## 🧠 Supported Gestures Out-of-the-Box

### 🔤 Alphabet Signs:
- `A`, `B`, `C`, `D`, `E`, `I`, `L`, `O`, `V`, `W`, `Y`

### 🗣️ Common Terms & Daily Communication Gestures:
- **`Help / Good`**: Thumbs Up (closed fist, thumb pointing straight UP)
- **`Bad`**: Thumbs Down (closed fist, thumb pointing straight DOWN)
- **`Food / Eat`**: All fingertips bunched together touching thumb tip (eating gesture)
- **`Water`**: "W" sign (three fingers upright & spread apart)
- **`Stop`**: Flat open hand with all 5 fingers upright & rigid facing the camera
- **`Please`**: Flat hand with 4 fingers straight together, thumb close to palm
- **`Sorry`**: Closed fist with thumb across front of knuckles
- **`Okay`**: Thumb & index tips touching in "OK" circle, other 3 fingers upright
- **`Call Me`**: Thumb & pinky extended ("phone" gesture), middle 3 fingers curled
- **`Rock On`**: Index & pinky extended, middle 2 curled, thumb folded
- **`You`**: Index finger pointing directly forward
- **`Hello`**: Open flat palm with all 5 fingers spread naturally
- **`Thank You`**: Flat open hand moving outward from chin/face
- **`Yes`**: Closed fist held upright
- **`No`**: Index & middle fingers pinched together onto thumb tip
- **`I Love You`**: Thumb, index, and pinky extended simultaneously
- **`Peace`**: Index & middle fingers extended in a separated "V" shape

*(Plus the ability to add and train any custom gesture dynamically via the Data Collection Studio!)*
