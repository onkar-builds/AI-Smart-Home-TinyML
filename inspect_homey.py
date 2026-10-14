
import joblib
import numpy as np
import wave
from python_speech_features import mfcc

model = joblib.load("models/homey_mfcc.joblib")

def extract(path):
    with wave.open(path, "rb") as audio:
        rate = audio.getframerate()
        signal = np.frombuffer(
            audio.readframes(audio.getnframes()),
            dtype=np.int16
        ).astype(np.float32) / 32768.0

    m = mfcc(
        signal, samplerate=rate,
        winlen=0.025, winstep=0.010,
        numcep=13, nfilt=26, nfft=512,
        appendEnergy=True
    )
    return np.concatenate([m.mean(axis=0), m.std(axis=0)]).reshape(1, -1)

for path in [
    "dataset_v2/LIGHT_ON/LIGHT_ON_01.wav",
    "dataset_v2/LIGHT_OFF/LIGHT_OFF_01.wav",
    "dataset_v2/FAN_ON/FAN_ON_01.wav",
    "dataset_v2/FAN_OFF/FAN_OFF_01.wav",
    "homey_mic_test.wav"
]:
    x = extract(path)
    prediction = model.predict(x)[0]
    print("\n", path)
    print("Prediction:", prediction)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)[0]
        for label, score in zip(model.classes_, scores):
            print(f"  {label}: {score:.3f}")
