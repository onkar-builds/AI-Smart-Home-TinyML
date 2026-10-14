
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

def extract_features(path):
    with wave.open(path, "rb") as audio:
        rate = audio.getframerate()
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        signal = np.frombuffer(
            audio.readframes(audio.getnframes()), dtype=np.int16
        ).astype(np.float32)

    if rate != 16000 or channels != 1 or width != 2:
        raise ValueError(f"Unexpected audio format: {path}")

    if len(signal) == 0:
        raise ValueError(f"Empty audio file: {path}")

    signal /= 32768.0

    # MFCCs describe the changing spectral shape of speech.
    coefficients = mfcc(
        signal,
        samplerate=rate,
        winlen=0.025,
        winstep=0.010,
        numcep=13,
        nfilt=26,
        nfft=512,
        appendEnergy=True
    )

    # Summarize each MFCC over the whole recording.
    # This is a baseline; it does not preserve word order well.
    features = np.concatenate([
        np.mean(coefficients, axis=0),
        np.std(coefficients, axis=0)
    ])

    return features.astype(np.float32)

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

for command in COMMANDS:
    print(f"{command}: {int(np.sum(y == command))} recordings")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced")
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nMFCC baseline test accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 1), "%")
print("\n", classification_report(y_test, predictions, zero_division=0))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/homey_mfcc.joblib")
print("Saved: models/homey_mfcc.joblib")
