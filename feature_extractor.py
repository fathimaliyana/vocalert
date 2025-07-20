import librosa
import numpy as np

def extract_features(audio, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    energy = np.mean(librosa.feature.rms(y=audio))
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitches = pitch[pitch > 0]
    pitch_value = np.mean(pitches) if pitches.size > 0 else 0

    if pitches.size > 1:
        jitter = np.mean(np.abs(np.diff(pitches))) / np.mean(pitches)
    else:
        jitter = 0

    shimmer = np.std(librosa.feature.rms(y=audio))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=4)
    mfccs_mean = np.mean(mfccs, axis=1)

    return [pitch_value, energy, jitter, shimmer, zcr, *mfccs_mean]
