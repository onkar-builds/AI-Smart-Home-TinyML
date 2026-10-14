
import os
import wave
import numpy as np
from glob import glob
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
        frames = audio.getnframes()
        raw = audio.readframes(frames)

    if rate != 16000 or channels != 1 or width != 2:
        raise ValueError(f"Unexpected audio format: {path}")

    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if len(signal) == 0:
        raise ValueError(f"Empty audio: {path}")

    signal /= 32768.0

    # Remove silence using short-time energy.
    frame_size = 400       # 25 ms at 16 kHz
    hop_size = 160         # 10 ms at 16 kHz
    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)))

    frames = np.lib.stride_tricks.sliding_window_view(
        signal, frame_size
    )[::hop_size]
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    active = frames[energy > max(0.008, energy.max() * 0.12)]

    if len(active) == 0:
        active = frames

    # Simple baseline features: 20 temporal energy segments.
    chunk_size = max(1, len(signal) // 20)
    rms = []
    for i in range(20):
        chunk = signal[i * chunk_size:(i + 1) * chunk_size]
        if len(chunk) == 0:
            rms.append(0.0)
        else:
            rms.append(float(np.sqrt(np.mean(chunk ** 2) + 1e-10)))

    # Add basic audio statistics.
    features = np.array(rms + [
        float(np.mean(np.abs(signal))),
        float(np.std(signal)),
        float(np.max(np.abs(signal))),
        float(len(signal) / rate)
    ], dtype=np.float32)

    return features

X, y = [], []

for command in COMMANDS:
    files = sorted(glob(os.path.join(DATASET, command, "*.wav")))
    if len(files) < 3:
        raise RuntimeError(f"Need at least 3 WAV files in {command}")

    for path in files:
        X.append(extract_features(path))
        y.append(command)

X = np.asarray(X)
y = np.asarray(y)

print("Total recordings:", len(y))
print("Feature shape:", X.shape)
print("\nRecordings per command:")
for command in COMMANDS:
    print(f"  {command}: {int(np.sum(y == command))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced")
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nPilot test accuracy:", round(accuracy_score(y_test, predictions) * 100, 1), "%")
print("\n", classification_report(y_test, predictions, zero_division=0))

os.makedirs("models", exist_ok=True)
import joblib
joblib.dump(model, "models/homey_baseline.joblib")
print("Saved baseline model: models/homey_baseline.joblib")
print("NOTE: This is an audio-energy baseline, not a reliable speech-recognition model.")
