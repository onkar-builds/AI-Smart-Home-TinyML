
import os
import wave
import joblib
import numpy as np
from glob import glob
from python_speech_features import mfcc
from sklearn.metrics import confusion_matrix, classification_report

DATASET = "dataset_v2"
COMMANDS = ["LIGHT_ON", "LIGHT_OFF", "FAN_ON", "FAN_OFF"]
MODEL = joblib.load("models/homey_mfcc.joblib")

def features(path):
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
    return np.concatenate([m.mean(axis=0), m.std(axis=0)])

actual, predicted = [], []

for command in COMMANDS:
    for path in sorted(glob(os.path.join(DATASET, command, "*.wav"))):
        actual.append(command)
        predicted.append(MODEL.predict(features(path).reshape(1, -1))[0])

print("TRAINING-DATA CHECK (not an independent accuracy test)")
print(classification_report(actual, predicted, labels=COMMANDS, zero_division=0))
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(actual, predicted, labels=COMMANDS))
