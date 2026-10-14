
import wave
import joblib
import numpy as np
from python_speech_features import mfcc

RATE = 16000
MODEL_PATH = "models/homey_mfcc.joblib"

def extract_features(signal):
    signal = signal.astype(np.float32) / 32768.0

    # Trim quiet audio at the beginning and end
    threshold = max(0.015, np.max(np.abs(signal)) * 0.08)
    active = np.where(np.abs(signal) > threshold)[0]

    if len(active) > 0:
        start = max(0, active[0] - int(0.10 * RATE))
        end = min(len(signal), active[-1] + int(0.10 * RATE))
        signal = signal[start:end]

    # Avoid processing an empty recording
    if len(signal) < int(0.25 * RATE):
        raise ValueError("Too little speech detected. Please record again.")

    features = mfcc(
        signal,
        samplerate=RATE,
        winlen=0.025,
        winstep=0.010,
        numcep=13,
        nfilt=26,
        nfft=512,
        appendEnergy=True
    )

    return np.concatenate(
        [np.mean(features, axis=0), np.std(features, axis=0)]
    ).reshape(1, -1)

with wave.open("homey_mic_test.wav", "rb") as f:
    if f.getframerate() != RATE or f.getnchannels() != 1:
        raise ValueError("Audio must be 16 kHz mono.")
    audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

model = joblib.load(MODEL_PATH)
prediction = model.predict(extract_features(audio))[0]

print("HOMEY prediction:", prediction)
