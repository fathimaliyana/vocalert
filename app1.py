import streamlit as st
import numpy as np
import time
import joblib
import base64
from io import BytesIO
import soundfile as sf
from streamlit_audio_recorder import audio_recorder
from feature_extractor import extract_features

# Load model
model = joblib.load("svm_model.pkl")

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
st.markdown("Record a 3-second audio clip and get vocal strain detection.")

status = st.empty()

audio_bytes = audio_recorder()  # Browser-based audio recorder

if audio_bytes:
    status.info("🎙️ Processing audio...")
    audio_np, samplerate = sf.read(BytesIO(audio_bytes))

    features = extract_features(audio_np, samplerate)

    if len(features) != model.n_features_in_:
        status.error(f"❌ Feature mismatch: Expected {model.n_features_in_}, got {len(features)}")
    else:
        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]

        if prediction == 1:
            status.error("⚠️ Vocal strain detected! Please rest your voice.")
            beep_audio, samplerate = generate_beep()
            html = get_beep_html(beep_audio, samplerate)
            st.markdown(html, unsafe_allow_html=True)
        else:
            status.success("✅ Voice is normal.")
else:
    status.info("🎙️ Click the record button above and speak for about 3 seconds.")
