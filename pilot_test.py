
import os
import random
import wave
import numpy as np

FOLDER = os.path.join("dataset", "LIGHT_ON")
random.seed(42)

files = sorted(
    f for f in os.listdir(FOLDER)
    if f.lower().endswith(".wav")
    and f.lower() != "light_on_test.wav"
)

if len(files) < 5:
    raise SystemExit("Need at least 5 recordings.")

random.shuffle(files)
train_files = files[:-4]
test_files = files[-4:]


def extract_features(path):
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        audio = np.frombuffer(
            wav.readframes(wav.getnframes()),
            dtype=np.int16
        ).astype(np.float32)

    if rate != 16000:
        raise ValueError(f"Expected 16000 Hz: {path}")

    if len(audio) == 0:
        raise ValueError(f"Empty audio: {path}")

    # Normalize volume and divide audio into 20 chunks.
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    chunks = np.array_split(audio, 20)

    return np.array([
        np.sqrt(np.mean(chunk ** 2) + 1e-10)
        for chunk in chunks
    ], dtype=np.float32)


train_features = np.array([
    extract_features(os.path.join(FOLDER, f))
    for f in train_files
])

test_features = np.array([
    extract_features(os.path.join(FOLDER, f))
    for f in test_files
])

reference = train_features.mean(axis=0)

print("Training recordings:", len(train_files))
print("Test recordings:", len(test_files))
print("\nDistance from LIGHT_ON reference:")

for filename, features in zip(test_files, test_features):
    distance = np.linalg.norm(features - reference)
    print(f"{filename}: {distance:.4f}")

print("\nAudio feature test complete.")
print("This is a pipeline check, not command-recognition accuracy.")
