import os
import numpy as np
import librosa

DATASET_PATH = "dataset"

COMMANDS = [
    "LIGHT_ON",
    "LIGHT_OFF",
    "FAN_ON",
    "FAN_OFF"
]

SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 13


def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    return mfcc


X = []
y = []

for label, command in enumerate(COMMANDS):

    folder = os.path.join(DATASET_PATH, command)

    if not os.path.exists(folder):
        print(f"Skipping {command}: folder not found")
        continue

    for filename in os.listdir(folder):

        if filename.endswith(".wav"):

            file_path = os.path.join(folder, filename)

            features = extract_features(file_path)

            X.append(features)
            y.append(label)

            print(f"Processed: {file_path}")


X = np.array(X)
y = np.array(y)

print("\nPreprocessing complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of samples:", len(X))