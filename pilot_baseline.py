
import os
import wave
import numpy as np

FOLDER = os.path.join("pilot_dataset", "LIGHT_ON")

files = sorted(
    f for f in os.listdir(FOLDER)
    if f.lower().endswith(".wav")
)

if len(files) < 5:
    raise SystemExit("Need at least 5 recordings.")

def load_audio(path):
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        audio = np.frombuffer(
            wav.readframes(wav.getnframes()),
            dtype=np.int16
        ).astype(np.float32)

    if rate != 16000 or channels != 1:
        raise ValueError(f"Expected 16 kHz mono audio: {path}")

    audio /= max(float(np.max(np.abs(audio))), 1.0)
    return audio

def features(audio):
    # Divide each recording into 20 time segments.
    segments = np.array_split(audio, 20)

    return np.array([
        np.sqrt(np.mean(segment ** 2) + 1e-10)
        for segment in segments
    ], dtype=np.float32)

vectors = []

for filename in files:
    audio = load_audio(os.path.join(FOLDER, filename))
    vectors.append(features(audio))

vectors = np.array(vectors)

reference = vectors.mean(axis=0)

print("LIGHT_ON recordings:", len(files))
print("Feature vectors created:", len(vectors))
print("\nDistance from average LIGHT_ON recording:")

for filename, vector in zip(files, vectors):
    distance = np.linalg.norm(vector - reference)
    print(f"{filename}: {distance:.4f}")

print("\nBaseline complete.")
print("These distances are not command-recognition accuracy.")
