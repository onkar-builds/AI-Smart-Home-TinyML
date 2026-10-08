import sounddevice as sd
import numpy as np
from python_speech_features import mfcc

SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 13

COMMANDS = [
    "LIGHT_ON",
    "LIGHT_OFF",
    "FAN_ON",
    "FAN_OFF"
]


def extract_features(audio):

    features = mfcc(
        audio,
        samplerate=SAMPLE_RATE,
        numcep=N_MFCC
    )

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8

    features = (features - mean) / std

    return features.astype(np.float32)


def sequence_distance(a, b):

    # Both recordings normally have the same number of frames.
    # Compare frame-by-frame MFCC patterns.
    frames = min(len(a), len(b))

    a = a[:frames]
    b = b[:frames]

    return np.mean(np.sqrt(np.sum((a - b) ** 2, axis=1)))


print("Get ready...")
input("Press ENTER and say your command...")

print("Recording...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

audio = audio.flatten()

print("Recording complete!")

features = extract_features(audio)

training_features = np.load("training_features.npy")
training_labels = np.load("training_labels.npy")

distances = []

for train_features, label in zip(
    training_features,
    training_labels
):

    distance = sequence_distance(
        features,
        train_features
    )

    distances.append((distance, label))


# Sort by closest recordings
distances.sort(key=lambda x: x[0])

# Take the 5 closest recordings
top_5 = distances[:5]

votes = {}

for distance, label in top_5:

    votes[label] = votes.get(label, 0) + 1


prediction = max(
    votes,
    key=votes.get
)


print("\n-----------------------------")
print("PREDICTED COMMAND:")
print(COMMANDS[prediction])
print("-----------------------------")

print("\nClosest recordings:")

for distance, label in top_5:

    print(
        f"{COMMANDS[label]}: "
        f"{distance:.2f}"
    )