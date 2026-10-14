
import os
import wave
import joblib
import numpy as np
from glob import glob
from python_speech_features import mfcc

model = joblib.load("models/homey_mfcc.joblib")
commands = ["LIGHT_ON", "LIGHT_OFF", "FAN_ON", "FAN_OFF"]

def features(path):
    with wave.open(path, "rb") as audio:
        signal = np.frombuffer(
            audio.readframes(audio.getnframes()),
            dtype=np.int16
        ).astype(np.float32) / 32768.0
        rate = audio.getframerate()

    m = mfcc(
        signal, samplerate=rate,
        winlen=0.025, winstep=0.010,
        numcep=13, nfilt=26, nfft=512,
        appendEnergy=True
    )
    return np.concatenate([m.mean(axis=0), m.std(axis=0)])

for command in commands:
    files = sorted(glob(os.path.join("dataset_v2", command, "*.wav")))
    print(f"\n{command}:")
    for path in files[:3]:
        prediction = model.predict(features(path).reshape(1, -1))[0]
        print(os.path.basename(path), "->", prediction)
