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

    # Normalize each MFCC coefficient
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8

    features = (features - mean) / std

    return features.astype(np.float32)


training_data = []
training_labels = []

print("Loading training data...\n")

for label, command in enumerate(COMMANDS):

    folder = os.path.join(DATASET_PATH, command)

    for filename in sorted(os.listdir(folder)):

        if filename.endswith(".wav"):

            path = os.path.join(folder, filename)

            features = extract_features(path)

            training_data.append(features)
            training_labels.append(label)

            print("Loaded:", path)


training_data = np.array(training_data, dtype=np.float32)
training_labels = np.array(training_labels)

np.save("training_features.npy", training_data)
np.save("training_labels.npy", training_labels)

print("\nTraining data prepared!")
print("Samples:", len(training_data))
print("Feature shape:", training_data.shape)
print("Saved:")
print("training_features.npy")
print("training_labels.npy")