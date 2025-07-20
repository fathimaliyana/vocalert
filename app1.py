import streamlit as st
import numpy as np
import time
import joblib
import base64
from io import BytesIO
import soundfile as sf
import sounddevice as sd
from feature_extractor import extract_features

# Load model
model = joblib.load("svm_model.pkl")

# Web-compatible beep sound (440Hz sine wave for 0.3s)
def generate_beep():
    fs = 22050
    duration = 0.3
    freq = 440
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio.astype(np.float32), fs

def get_beep_html(audio, samplerate):
    buffer = BytesIO()
    sf.write(buffer, audio, samplerate, format="WAV")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    """

st.set_page_config(page_title="Vocal Strain Monitor", layout="centered")
st.title("🎤 Vocal Strain Monitor (Web Compatible)")
st.markdown("This app records 3-second audio clips and alerts you if vocal strain is detected.")

status = st.empty()

# Session state
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

# Start and Stop buttons
col1, col2 = st.columns(2)
start_button = col1.button("▶️ Start Monitoring", key="start_button")
stop_button = col2.button("⏹️ Stop Monitoring", key="stop_button")

# Monitoring function
def monitor():
    duration = 3
    fs = 22050
    while st.session_state.monitoring:
        status.info("🎙️ Recording...")
        try:
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
        except Exception as e:
            status.error(f"🎙️ Recording failed: {e}")
            break

        audio = audio.flatten()
        features = extract_features(audio, fs)

        if len(features) != model.n_features_in_:
            status.error(f"❌ Feature mismatch: Expected {model.n_features_in_}, got {len(features)}")
            break

        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]

        if prediction == 1:
            status.error("⚠️ Vocal strain detected! Please rest your voice.")
            beep_audio, samplerate = generate_beep()
            html = get_beep_html(beep_audio, samplerate)
            st.markdown(html, unsafe_allow_html=True)
        else:
            status.success("✅ Voice is normal.")

        time.sleep(0.5)

# Handle buttons
if start_button:
    st.session_state.monitoring = True
    monitor()

if stop_button:
    st.session_state.monitoring = False
    status.warning("🛑 Monitoring stopped.")
