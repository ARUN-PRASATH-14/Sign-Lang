"""
Smart Communication System for Deaf and Dumb Using AI-Based Sign Language Translation
Streamlit Interactive Web Application.
"""

import base64
import json
import os
import time
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from models.architecture import SignCNNLSTM, SignLandmarkMLP
from utils.hand_detector import HandDetector
from utils.tts_engine import speak_text
from utils.stt_engine import listen_once
from utils.gesture_recognizer import BiomechanicalGestureRecognizer

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Sign Language Translation Hub",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    * {
        font-family: 'Outfit', sans-serif;
    }

    /* Overall Background and Themes */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0a0e17 0%, #0f172a 100%);
        color: #f1f5f9;
    }

    /* Academic Header Card */
    .academic-header {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(12px);
    }
    .academic-header h1 {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #60a5fa;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .academic-header h2 {
        font-size: 15px;
        font-weight: 600;
        color: #93c5fd;
        margin-bottom: 4px;
    }
    .academic-header p {
        font-size: 13px;
        color: #cbd5e1;
        margin: 2px 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.4);
        margin-right: 8px;
    }

    /* Glassmorphism Metric Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
    }

    .prediction-banner {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border-radius: 14px;
        padding: 18px 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.35);
        margin-bottom: 16px;
        animation: pulse-border 2s infinite;
    }
    .prediction-sign {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 1px;
        line-height: 1.1;
    }
    .prediction-meta {
        font-size: 14px;
        opacity: 0.9;
        font-weight: 500;
    }

    .sentence-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        min-height: 70px;
        font-size: 20px;
        color: #f8fafc;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.4);
    }

    .deaf-display-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 2px solid #818cf8;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        color: white;
        margin-top: 16px;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
    }
    .deaf-display-text {
        font-size: 38px;
        font-weight: 800;
        color: #e0e7ff;
        margin: 10px 0;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
        color: #94a3b8;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.2) !important;
        color: #60a5fa !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# AUDIO WEB SPEECH SCRIPT HELPER
# ==========================================
def browser_tts_html(text: str):
    """Executes client-side Web Speech API synthesis in the user's browser."""
    clean_text = text.replace("'", "\\'").replace('"', '\\"')
    return f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance("{clean_text}");
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        }}
    }})();
    </script>
    """


# ==========================================
# MODEL LOADER
# ==========================================
@st.cache_resource
def load_models_and_labels():
    encoder_path = os.path.join("models", "label_encoder.json")
    if not os.path.exists(encoder_path):
        from train_initial_model import train_models

        train_models()

    with open(encoder_path, "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # Load MLP model
    mlp_model = SignLandmarkMLP(input_dim=63, num_classes=num_classes)
    mlp_path = os.path.join("models", "sign_model.pth")
    if os.path.exists(mlp_path):
        mlp_model.load_state_dict(torch.load(mlp_path, map_location="cpu"))
    mlp_model.eval()

    # Load CNN-LSTM model
    cnn_lstm_model = SignCNNLSTM(input_dim=63, num_classes=num_classes, hidden_dim=64)
    cl_path = os.path.join("models", "sign_model_cnn_lstm.pth")
    if os.path.exists(cl_path):
        try:
            cnn_lstm_model.load_state_dict(torch.load(cl_path, map_location="cpu"))
        except Exception:
            pass
    cnn_lstm_model.eval()

    return label_map, mlp_model, cnn_lstm_model


label_map, mlp_model, cnn_lstm_model = load_models_and_labels()


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "current_sentence" not in st.session_state:
    st.session_state.current_sentence = []
if "recognized_history" not in st.session_state:
    st.session_state.recognized_history = []
if "reverse_text" not in st.session_state:
    st.session_state.reverse_text = "Waiting for speech or text input..."
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None


# ==========================================
# TOP ACADEMIC HEADER (Matches Slide 1)
# ==========================================
st.markdown(
    """
    <div class="academic-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <span class="badge-pill">Autonomous Institution</span>
                <span class="badge-pill">Anna University Affiliated</span>
                <span class="badge-pill">IoT & AI Dept</span>
                <h1>Smart Communication System for Deaf and Dumb</h1>
                <h2>AI-Based Real-Time Sign Language Translation & Two-Way Communication</h2>
                <p><strong>Institution:</strong> K.L.N. College of Engineering, Dept. of Computer Science & Engineering (IoT)</p>
            </div>
            <div style="text-align: right; background: rgba(15, 23, 42, 0.6); padding: 12px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);">
                <p style="color: #60a5fa; font-weight: 700; margin: 0;">GUIDE: Mrs. Roselin Reeta P</p>
                <p style="font-size: 11px; color: #94a3b8; margin: 0 0 6px 0;">Assistant Professor 2</p>
                <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px;">
                    <p style="font-size: 11px; color: #e2e8f0; margin: 0;"><strong>Team:</strong> Joshua J [910623110021]</p>
                    <p style="font-size: 11px; color: #e2e8f0; margin: 0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Hazi Khan U [910623110016]</p>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Play speech script if requested
if st.session_state.audio_to_play:
    st.components.v1.html(
        browser_tts_html(st.session_state.audio_to_play), height=0, width=0
    )
    st.session_state.audio_to_play = None


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=500&auto=format&fit=crop&q=60",
        caption="AI Vision Sign Language Pipeline",
        use_container_width=True,
    )
    st.markdown("### ⚙️ System Settings")

    selected_model_type = st.radio(
        "Classifier Architecture",
        options=["Fast Landmark MLP", "Spatial 1D-CNN + LSTM"],
        index=0,
        help="Choose between ultra-low latency MLP or Hybrid CNN-LSTM temporal network.",
    )

    conf_threshold = st.slider(
        "Detection Confidence",
        min_value=0.40,
        max_value=0.95,
        value=0.65,
        step=0.05,
    )

    cam_id = st.selectbox("Webcam Index", options=[0, 1, 2], index=0)

    st.markdown("---")
    st.markdown("### 🎙️ Audio Settings (TTS)")
    auto_speak = st.checkbox(
        "Auto-Speak Confirmed Signs",
        value=True,
        help="Automatically speaks gestures when held stably.",
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.write(f"**Recognized Classes:** {len(label_map)}")
    st.caption(", ".join(list(label_map.values())[:10]) + "...")


# ==========================================
# TABS NAVIGATION
# ==========================================
tab_live, tab_reverse, tab_studio, tab_train, tab_dict, tab_arch = st.tabs(
    [
        "📹 Live Sign Translator",
        "🗣️ Speech to Deaf/Mute (Reverse Flow)",
        "📸 Data Collection Studio",
        "🧠 Model Training & Analytics",
        "📚 Sign Dictionary & Guide",
        "🏛️ Architecture & Project Specs",
    ]
)


# ==========================================
# TAB 1: LIVE SIGN TRANSLATOR
# ==========================================
with tab_live:
    col_video, col_panel = st.columns([7, 5])

    with col_video:
        st.subheader("Webcam Sign Language Capture")

        capture_mode = st.radio(
            "Select Camera Input Mode:",
            options=["📸 Browser Camera (Streamlit Cloud & Mobile)", "🔴 Continuous Live Stream (Local)", "📁 Upload Image"],
            horizontal=True,
            help="Browser Camera works on Streamlit Cloud, phones, tablets, and laptops."
        )

        video_placeholder = st.empty()
        status_placeholder = st.empty()

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🔊 Speak Entire Sentence", use_container_width=True):
                if st.session_state.current_sentence:
                    full_txt = " ".join(st.session_state.current_sentence)
                    speak_text(full_txt)
                    st.session_state.audio_to_play = full_txt
                    st.rerun()
        with col_c2:
            if st.button("🧹 Clear Sentence", use_container_width=True):
                st.session_state.current_sentence.clear()
                st.rerun()

    with col_panel:
        st.subheader("Live Translation Feed")

        live_banner = st.empty()
        sentence_display = st.empty()

        live_banner.markdown(
            """
            <div class="prediction-banner" style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%);">
                <div class="prediction-sign">--</div>
                <div class="prediction-meta">Camera feed is paused. Toggle 'Start Live Translation'</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sentence_str = (
            " ".join(st.session_state.current_sentence)
            if st.session_state.current_sentence
            else "[No signs captured yet]"
        )
        sentence_display.markdown(
            f"""
            <div class="glass-card">
                <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Constructed Sentence:</span>
                <div class="sentence-box" style="margin-top: 8px;">{sentence_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Real-Time Gesture Confidence")
        progress_bar = st.progress(0.0)
        conf_metric = st.empty()

    # Model and Detector Setup
    detector = HandDetector(mode=False, max_hands=2, detection_con=0.7, track_con=0.6)
    recognizer = BiomechanicalGestureRecognizer()
    active_model = (
        cnn_lstm_model
        if selected_model_type == "Spatial 1D-CNN + LSTM"
        else mlp_model
    )

    common_terms_speak = [
        "Hello", "Thank You", "Please", "Sorry", "Yes", "No",
        "I Love You", "Peace", "Help / Good", "Bad", "Water",
        "Food / Eat", "Okay", "Stop", "Call Me", "Rock On", "You"
    ]

    def process_and_display_frame(frame_input):
        h_f, w_f, _ = frame_input.shape
        frame_proc, results = detector.find_hands(frame_input.copy(), draw=True)

        found_sign = None
        found_conf = 0.0
        found_hand = "Right"

        if results and results.multi_hand_landmarks:
            for idx_h, hand_lms in enumerate(results.multi_hand_landmarks):
                lm_list, bbox, handedness = detector.get_hand_landmarks(frame_proc, idx_h)
                features = detector.extract_normalized_features(hand_lms)
                raw_lms = detector.get_raw_landmarks(idx_h)
                geo_res = recognizer.analyze_hand(raw_lms) if raw_lms is not None else None

                if features is not None and bbox is not None:
                    t_in = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = active_model(t_in)
                        probs = F.softmax(logits, dim=1).numpy()[0]
                        idx_max = int(np.argmax(probs))
                        c_max = float(probs[idx_max])
                        ml_sign = label_map.get(str(idx_max), "Unknown")

                    if geo_res and geo_res["confidence"] >= 0.80:
                        pred_s = geo_res["label"]
                        conf_s = geo_res["confidence"]
                    elif c_max >= conf_threshold and ml_sign != "Unknown":
                        pred_s = ml_sign
                        conf_s = c_max
                    elif geo_res:
                        pred_s = geo_res["label"]
                        conf_s = geo_res["confidence"]
                    else:
                        pred_s = "Unknown"
                        conf_s = 0.0

                    if pred_s and conf_s >= 0.55 and pred_s != "Unknown":
                        found_sign = pred_s
                        found_conf = conf_s
                        found_hand = handedness

                        x1, y1, x2, y2 = bbox
                        pad = 20
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(w_f, x2 + pad), min(h_f, y2 + pad)

                        cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame_proc,
                            f"{handedness}: {pred_s}",
                            (x1, max(30, y1 - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame_proc,
                            f"{conf_s*100:.1f}%",
                            (x1, max(30, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2,
                        )

        return frame_proc, found_sign, found_conf, found_hand

    # --- MODE 1: BROWSER CAMERA (IDEAL FOR STREAMLIT CLOUD & MOBILE) ---
    if "Browser Camera" in capture_mode:
        status_placeholder.info("📸 Click below to take a photo of your hand sign. Works directly on Streamlit Cloud & phones!")
        cam_photo = st.camera_input("Capture Sign Photo")

        if cam_photo is not None:
            bytes_data = cam_photo.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            processed_img, p_sign, p_conf, p_hand = process_and_display_frame(cv_img)

            if p_sign:
                live_banner.markdown(
                    f"""
                    <div class="prediction-banner">
                        <div class="prediction-sign">{p_sign}</div>
                        <div class="prediction-meta">{p_hand} Hand | Confidence: {p_conf*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                progress_bar.progress(min(1.0, float(p_conf)))
                conf_metric.caption(f"Detected Sign: **{p_sign}** ({p_conf*100:.1f}%)")

                if not st.session_state.current_sentence or st.session_state.current_sentence[-1] != p_sign:
                    st.session_state.current_sentence.append(p_sign)
                    st.session_state.recognized_history.append(
                        {"sign": p_sign, "time": time.strftime("%H:%M:%S")}
                    )

                sentence_str = " ".join(st.session_state.current_sentence)
                sentence_display.markdown(
                    f"""
                    <div class="glass-card">
                        <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Constructed Sentence:</span>
                        <div class="sentence-box" style="margin-top: 8px;">{sentence_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if auto_speak and p_sign in common_terms_speak:
                    clean_spoken = p_sign.replace(" / ", " or ")
                    speak_text(clean_spoken)
                    st.session_state.audio_to_play = clean_spoken
            else:
                st.warning("No clear hand detected in photo. Please ensure good lighting and face your hand to the camera.")

            # Display annotated result
            frame_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # --- MODE 2: UPLOAD IMAGE ---
    elif "Upload Image" in capture_mode:
        status_placeholder.info("📁 Upload an image of a hand sign to translate.")
        uploaded_file = st.file_uploader("Choose a hand gesture photo (JPG / PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            processed_img, p_sign, p_conf, p_hand = process_and_display_frame(cv_img)

            if p_sign:
                live_banner.markdown(
                    f"""
                    <div class="prediction-banner">
                        <div class="prediction-sign">{p_sign}</div>
                        <div class="prediction-meta">{p_hand} Hand | Confidence: {p_conf*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                progress_bar.progress(min(1.0, float(p_conf)))
                conf_metric.caption(f"Detected Sign: **{p_sign}** ({p_conf*100:.1f}%)")

                if not st.session_state.current_sentence or st.session_state.current_sentence[-1] != p_sign:
                    st.session_state.current_sentence.append(p_sign)

                sentence_str = " ".join(st.session_state.current_sentence)
                sentence_display.markdown(
                    f"""
                    <div class="glass-card">
                        <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Constructed Sentence:</span>
                        <div class="sentence-box" style="margin-top: 8px;">{sentence_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if auto_speak and p_sign in common_terms_speak:
                    clean_spoken = p_sign.replace(" / ", " or ")
                    speak_text(clean_spoken)
                    st.session_state.audio_to_play = clean_spoken
            else:
                st.warning("No hand landmarks recognized in uploaded image.")

            frame_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # --- MODE 3: CONTINUOUS LOCAL STREAM ---
    elif "Continuous Live Stream" in capture_mode:
        run_live = st.toggle("🔴 Start Local Live Camera Stream", value=False)

        if run_live:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                status_placeholder.error(
                    "⚠️ Local webcam not available. If you are on Streamlit Cloud, switch to '📸 Browser Camera' above!"
                )
            else:
                stable_sign = ""
                stable_count = 0
                threshold_frames = 8
                status_placeholder.info("🟢 Local camera feed running.")

                while run_live:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.error("Could not read frame from local camera.")
                        break

                    frame = cv2.flip(frame, 1)
                    processed_img, p_sign, p_conf, p_hand = process_and_display_frame(frame)

                    if p_sign:
                        live_banner.markdown(
                            f"""
                            <div class="prediction-banner">
                                <div class="prediction-sign">{p_sign}</div>
                                <div class="prediction-meta">{p_hand} Hand | Confidence: {p_conf*100:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        progress_bar.progress(min(1.0, float(p_conf)))
                        conf_metric.caption(f"Detected Sign: **{p_sign}** ({p_conf*100:.1f}%)")

                        if p_sign == stable_sign:
                            stable_count += 1
                        else:
                            stable_sign = p_sign
                            stable_count = 1

                        if stable_count == threshold_frames:
                            if not st.session_state.current_sentence or st.session_state.current_sentence[-1] != p_sign:
                                st.session_state.current_sentence.append(p_sign)

                                sentence_str = " ".join(st.session_state.current_sentence)
                                sentence_display.markdown(
                                    f"""
                                    <div class="glass-card">
                                        <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Constructed Sentence:</span>
                                        <div class="sentence-box" style="margin-top: 8px;">{sentence_str}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                if auto_speak and p_sign in common_terms_speak:
                                    clean_spoken = p_sign.replace(" / ", " or ")
                                    speak_text(clean_spoken)
                    else:
                        stable_count = 0
                        progress_bar.progress(0.0)

                    frame_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    time.sleep(0.03)

                cap.release()


# ==========================================
# TAB 2: REVERSE COMMUNICATION (SPEECH TO TEXT/SIGNS)
# ==========================================
with tab_reverse:
    st.subheader("Reverse Communication Flow: Hearing Person ➡️ Deaf/Mute Individual")
    st.write(
        "Normal individuals speak naturally into the microphone or enter text. "
        "The AI translates it into large, crystal-clear visual captions and displays sign spelling illustrations for the deaf user."
    )

    col_rev_in, col_rev_out = st.columns([5, 7])

    with col_rev_in:
        st.markdown("#### 1. Normal/Hearing User Input")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🎙️ Speak via Microphone", use_container_width=True):
                with st.spinner("Listening for 5 seconds... Speak now into microphone."):
                    text, status_msg, success = listen_once(timeout=5)
                    if success:
                        st.session_state.reverse_text = text
                        st.success(f"Recognized: '{text}'")
                    else:
                        st.warning(status_msg)
        with col_b2:
            preset = st.selectbox(
                "Quick Phrases",
                options=[
                    "",
                    "Hello, how can I help you?",
                    "Where are you going?",
                    "Nice to meet you",
                    "Do you need water or food?",
                    "Please wait a moment",
                ],
            )
            if preset:
                st.session_state.reverse_text = preset

        typed_input = st.text_input(
            "Or Type Message for Deaf Individual:",
            value="" if st.session_state.reverse_text.startswith("Waiting") else st.session_state.reverse_text,
            placeholder="Type anything here...",
        )
        if st.button("Update Display for Deaf User", use_container_width=True):
            if typed_input.strip():
                st.session_state.reverse_text = typed_input.strip()
                st.rerun()

    with col_rev_out:
        st.markdown("#### 2. Visual Display Screen (For Deaf Individual)")
        st.markdown(
            f"""
            <div class="deaf-display-card">
                <span style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #a5b4fc; font-weight: 700;">Spoken Words Converted To Text</span>
                <div class="deaf-display-text">"{st.session_state.reverse_text}"</div>
                <div style="font-size: 13px; color: #cbd5e1; margin-top: 8px;">Displayed in High Contrast for Instant Comprehension</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 3. Sign Language Finger-Spelling Guide")
        # Break words into letters and show letter representations
        words = st.session_state.reverse_text.upper().split()
        if words and not st.session_state.reverse_text.startswith("Waiting"):
            clean_letters = [char for char in st.session_state.reverse_text.upper() if char.isalpha()][:15]
            if clean_letters:
                badge_html = " ".join([f"<span class='badge-pill' style='font-size: 18px; padding: 6px 14px;'>{ch}</span>" for ch in clean_letters])
                st.markdown(f"<div style='margin-top: 12px;'>{badge_html}</div>", unsafe_allow_html=True)


# ==========================================
# TAB 3: DATA COLLECTION STUDIO
# ==========================================
with tab_studio:
    st.subheader("Data Collection & Custom Sign Recording")
    st.write("Capture new gesture images directly from your camera to expand the vocabulary of the AI model.")

    col_d1, col_d2 = st.columns([5, 7])

    with col_d1:
        st.markdown("#### Record New Gesture")
        new_gesture_name = st.text_input("New Gesture Name (e.g., 'Namaste', 'Water', 'Z')", value="")
        sample_target = st.slider("Target Number of Samples", min_value=10, max_value=100, value=30, step=5)

        st.info("Tip: You can also run `python dataset_collector.py` in your terminal for high-speed continuous capture!")

        uploaded_img = st.file_uploader("Or Upload Custom Gesture Image", type=["jpg", "jpeg", "png"])
        if uploaded_img and new_gesture_name:
            if st.button("Save Uploaded Image to Dataset"):
                target_folder = os.path.join("dataset", new_gesture_name.strip())
                os.makedirs(target_folder, exist_ok=True)
                img_path = os.path.join(target_folder, f"custom_{int(time.time())}.jpg")
                with open(img_path, "wb") as f:
                    f.write(uploaded_img.getbuffer())
                st.success(f"Saved image to dataset/{new_gesture_name.strip()}/")

    with col_d2:
        st.markdown("#### Current Dataset Folders")
        dataset_root = "dataset"
        if os.path.exists(dataset_root):
            folders = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
            if folders:
                folder_data = []
                for fld in folders:
                    count = len([f for f in os.listdir(os.path.join(dataset_root, fld)) if f.lower().endswith((".jpg", ".png"))])
                    folder_data.append({"Gesture": fld, "Image Count": count})
                st.dataframe(folder_data, use_container_width=True)
            else:
                st.write("No custom gesture folders recorded yet. Baseline synthetic dataset of 18 classes is loaded.")
        else:
            st.write("Dataset directory ready.")


# ==========================================
# TAB 4: MODEL TRAINING & METRICS
# ==========================================
with tab_train:
    st.subheader("Model Training Pipeline & Performance Analytics")
    st.write("Preprocess hand landmarks and train the CNN-LSTM or MLP neural network.")

    col_t1, col_t2 = st.columns([5, 7])

    with col_t1:
        st.markdown("#### Retrain Model")
        train_epochs = st.slider("Number of Epochs", min_value=10, max_value=60, value=30, step=5)
        train_model_choice = st.selectbox("Model to Train", ["SignLandmarkMLP", "SignCNNLSTM (Hybrid)"])

        if st.button("🚀 Start Training Pipeline", use_container_width=True):
            with st.spinner("Training model in progress..."):
                from train_cnn import train_pipeline

                m_type = "cnn_lstm" if "CNN" in train_model_choice else "mlp"
                history = train_pipeline(epochs=train_epochs, model_type=m_type)
                st.success("Training completed! Models updated successfully.")
                st.rerun()

    with col_t2:
        st.markdown("#### Model Training Curves")
        history_file = os.path.join("models", "training_history.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history_data = json.load(f)

            st.line_chart(
                {
                    "Train Loss": history_data.get("train_loss", []),
                    "Val Loss": history_data.get("val_loss", []),
                }
            )
            st.line_chart({"Val Accuracy (%)": history_data.get("val_accuracy", [])})
        else:
            st.info("No training history file found. Run training above to generate interactive loss and accuracy curves.")


# ==========================================
# TAB 5: SIGN DICTIONARY & GUIDE
# ==========================================
with tab_dict:
    st.subheader("Sign Language Dictionary & Biomechanical Reference")
    st.write("Explore standard signs, hand finger configurations, and MediaPipe landmark joint patterns.")

    dict_items = [
        {"Gesture": "Help / Good", "Type": "Common Term / Need", "Formation": "Thumbs Up! Closed fist with thumb extended vertically upright."},
        {"Gesture": "Bad", "Type": "Common Term", "Formation": "Thumbs Down! Closed fist with thumb pointing straight down."},
        {"Gesture": "Food / Eat", "Type": "Daily Need", "Formation": "All four fingertips bunched together touching thumb tip near chin/mouth."},
        {"Gesture": "Water", "Type": "Daily Need", "Formation": "'W' hand: Index, middle, and ring fingers upright and separated."},
        {"Gesture": "Stop", "Type": "Command", "Formation": "Flat open hand with all 5 fingers upright and rigid facing the camera."},
        {"Gesture": "Please", "Type": "Politeness", "Formation": "Flat hand with 4 fingers straight together and thumb resting against palm side."},
        {"Gesture": "Sorry", "Type": "Apology", "Formation": "Closed fist with thumb folded across knuckles in front of chest."},
        {"Gesture": "Okay", "Type": "Agreement", "Formation": "Thumb and index tips touching in 'OK' circle, other 3 fingers extended upright."},
        {"Gesture": "Call Me", "Type": "Action", "Formation": "Thumb and pinky extended sideways in 'phone' shape, middle 3 curled."},
        {"Gesture": "Rock On", "Type": "Slang / Cool", "Formation": "Index and pinky extended up, middle 2 curled down, thumb folded over middle."},
        {"Gesture": "You", "Type": "Pronoun / Direct", "Formation": "Index finger pointing forward towards the person/camera."},
        {"Gesture": "Hello", "Type": "Greeting", "Formation": "Open flat palm facing forward with 5 fingers spread naturally upright."},
        {"Gesture": "Thank You", "Type": "Politeness", "Formation": "Flat open hand with fingers together, moving outward from chin."},
        {"Gesture": "Yes", "Type": "Response", "Formation": "Closed fist held upright, nodding slightly forward."},
        {"Gesture": "No", "Type": "Response", "Formation": "Index and middle fingers snapped closed against thumb tip."},
        {"Gesture": "I Love You", "Type": "Expression", "Formation": "Thumb, index finger, and pinky finger extended simultaneously."},
        {"Gesture": "Peace", "Type": "Common Sign", "Formation": "Index and middle fingers extended apart in 'V' peace sign."},
        {"Gesture": "A", "Type": "Alphabet", "Formation": "Closed fist with thumb upright resting flat beside index finger."},
        {"Gesture": "B", "Type": "Alphabet", "Formation": "Four fingers straight and tightly together pointing upward, thumb folded across palm."},
        {"Gesture": "C", "Type": "Alphabet", "Formation": "Curved fingers and thumb forming an open 'C' shape."},
        {"Gesture": "D", "Type": "Alphabet", "Formation": "Index finger extended vertically, thumb touching middle, ring, pinky tips."},
        {"Gesture": "E", "Type": "Alphabet", "Formation": "All four fingers bent curled down, thumb tucked beneath fingertips."},
        {"Gesture": "I", "Type": "Alphabet", "Formation": "Pinky finger extended straight up, other three fingers closed in fist."},
        {"Gesture": "L", "Type": "Alphabet", "Formation": "Index finger pointing up, thumb extended horizontally at 90-degree 'L' angle."},
        {"Gesture": "O", "Type": "Alphabet", "Formation": "All five fingertips curved and touching together in an 'O' ring shape (Slide 5)."},
        {"Gesture": "V", "Type": "Alphabet", "Formation": "Index and middle fingers extended in a V shape."},
        {"Gesture": "W", "Type": "Alphabet", "Formation": "Index, middle, and ring fingers upright and separated."},
        {"Gesture": "Y", "Type": "Alphabet", "Formation": "Thumb and pinky fully extended sideways/upwards, middle 3 curled."},
    ]

    st.dataframe(dict_items, use_container_width=True)


# ==========================================
# TAB 6: ARCHITECTURE & TECHNICAL SPECS
# ==========================================
with tab_arch:
    st.subheader("System Architecture & Project Specifications")
    st.write("Detailed 5-Layer pipeline matching the project design of K.L.N. College of Engineering:")

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color: #60a5fa; margin-top: 0;">1. Input Layer</h4>
            <p>Captures live video feed at 30 FPS from standard USB / built-in webcam via OpenCV (<code>cv2.VideoCapture</code>).</p>

            <h4 style="color: #34d399;">2. Pre-Processing Layer</h4>
            <p>Utilizes Google MediaPipe Hands framework to isolate 21 3D hand keypoints (wrist, joints, fingertips) per hand with zero hardware sensors required.</p>

            <h4 style="color: #fbbf24;">3. Feature Extraction & Normalization Layer</h4>
            <p>Extracts 63 coordinate values per hand. Computes wrist-relative vector coordinates (<code>x - x0, y - y0, z - z0</code>) and normalizes by maximum Euclidean hand span. This guarantees full invariance to camera distance, user hand size, and position on screen.</p>

            <h4 style="color: #f87171;">4. Deep Learning Model Layer</h4>
            <p>Implements <strong>1D-CNN (Spatial feature extraction) + Bidirectional LSTM (Temporal dynamic sequence learning)</strong> and high-speed Deep Landmark MLP classifiers.</p>

            <h4 style="color: #a78bfa;">5. Output & Reverse Communication Layer</h4>
            <p><strong>Deaf ➡️ Hearing:</strong> Recognized text is synthesized into human speech via Text-to-Speech (TTS Engine).<br/>
            <strong>Hearing ➡️ Deaf:</strong> Microphone captures spoken audio, converts via Speech-to-Text (STT Engine) into visual high-contrast captions and sign finger-spelling representations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
