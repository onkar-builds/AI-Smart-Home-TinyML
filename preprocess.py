import os
import wave
import numpy as np
from python_speech_features import mfcc

DATASET_PATH = "dataset"

COMMANDS = [
    "LIGHT_ON",
    "LIGHT_OFF",
    "FAN_ON",
    "FAN_OFF"
]

SAMPLE_RATE = 16000
N_MFCC = 13


def extract_features(file_path):

    with wave.open(file_path, "rb") as wav_file:

        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)

    features = mfcc(
        audio,
        samplerate=sample_rate,
        numcep=N_MFCC
    )

    return features


X = []
y = []

for label, command in enumerate(COMMANDS):

    folder = os.path.join(DATASET_PATH, command)

    if not os.path.exists(folder):
        print(f"Skipping {command}: folder not found")
        continue

    for filename in sorted(os.listdir(folder)):

        if filename.endswith(".wav"):

            file_path = os.path.join(folder, filename)

            features = extract_features(file_path)

            X.append(features)
            y.append(label)

            print(f"Processed: {file_path}")


X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nPreprocessing complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of samples:", len(X))
np.save("X.npy", X)
np.save("y.npy", y)

print("Saved X.npy and y.npy")