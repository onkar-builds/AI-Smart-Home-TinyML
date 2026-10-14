
import os
import wave
import joblib
import numpy as np

from glob import glob
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATASET = "dataset_v2"
COMMANDS = ["LIGHT_ON", "LIGHT_OFF", "FAN_ON", "FAN_OFF"]
FRAMES = 100
COEFFICIENTS = 13

def extract_features(path):
    with wave.open(path, "rb") as audio:
        rate = audio.getframerate()
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        signal = np.frombuffer(
            audio.readframes(audio.getnframes()),
            dtype=np.int16
        ).astype(np.float32)

    if rate != 16000 or channels != 1 or width != 2:
        raise ValueError(f"Unexpected audio format: {path}")
    if len(signal) == 0:
        raise ValueError(f"Empty audio: {path}")

    signal /= 32768.0

    # Extract MFCCs over time.
    m = mfcc(
        signal,
        samplerate=rate,
        winlen=0.025,
        winstep=0.010,
        numcep=COEFFICIENTS,
        nfilt=26,
        nfft=512,
        appendEnergy=True
    )

    # Resample the sequence to a fixed number of frames.
    old_positions = np.linspace(0, 1, len(m))
    new_positions = np.linspace(0, 1, FRAMES)

    sequence = np.stack([
        np.interp(new_positions, old_positions, m[:, i])
        for i in range(COEFFICIENTS)
    ], axis=1)

    return sequence.flatten().astype(np.float32)

X, y = [], []

for command in COMMANDS:
    files = sorted(glob(os.path.join(DATASET, command, "*.wav")))
    if len(files) < 4:
        raise RuntimeError(f"Not enough recordings in {command}")

    for path in files:
        X.append(extract_features(path))
        y.append(command)

X = np.asarray(X)
y = np.asarray(y)

print("Total recordings:", len(y))
print("Feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced")
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nHeld-out test accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 1), "%")
print(classification_report(y_test, predictions, zero_division=0))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/homey_sequence.joblib")
print("\nSaved: models/homey_sequence.joblib")
