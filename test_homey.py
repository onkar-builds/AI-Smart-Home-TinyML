
import os
import wave
import tempfile
import numpy as np
import sounddevice as sd
import joblib
from python_speech_features import mfcc

MODEL_PATH = "models/homey_mfcc.joblib"
SAMPLE_RATE = 16000
DURATION = 3

def extract_features(signal):
    signal = signal.astype(np.float32) / 32768.0

    coefficients = mfcc(
        signal,
        samplerate=SAMPLE_RATE,
        winlen=0.025,
        winstep=0.010,
        numcep=13,
        nfilt=26,
        nfft=512,
        appendEnergy=True
    )

    return np.concatenate([
        np.mean(coefficients, axis=0),
        np.std(coefficients, axis=0)
    ]).astype(np.float32).reshape(1, -1)

model = joblib.load(MODEL_PATH)

print("HOMEY MFCC microphone test")
print("Speak one command clearly.")
input("Press Enter when ready...")

print("Recording for 3 seconds. Speak now!")
recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)
sd.wait()

signal = recording.flatten()
features = extract_features(signal)
prediction = model.predict(features)[0]

print("\nHOMEY prediction:", prediction)
print("This is a preliminary prototype; verify the prediction.")

