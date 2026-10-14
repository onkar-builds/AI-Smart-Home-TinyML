
import os
import wave
import joblib
import numpy as np
from python_speech_features import mfcc

RATE = 16000
MODEL_PATH = "models/homey_mfcc.joblib"
DATASET = "dataset_v2"

def extract_features(path):
    with wave.open(path, "rb") as f:
        audio = np.frombuffer(
            f.readframes(f.getnframes()), dtype=np.int16
        )

    signal = audio.astype(np.float32) / 32768.0

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

    return np.concatenate([
        np.mean(features, axis=0),
        np.std(features, axis=0)
    ]).reshape(1, -1)

model = joblib.load(MODEL_PATH)

for label in ["LIGHT_ON", "LIGHT_OFF", "FAN_ON", "FAN_OFF"]:
    folder = os.path.join(DATASET, label)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))

    print(f"\n{label}:")
    for filename in files[:5]:
        path = os.path.join(folder, filename)
        prediction = model.predict(extract_features(path))[0]
        print(f"  {filename} -> {prediction}")
