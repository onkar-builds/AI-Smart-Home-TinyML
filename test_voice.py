import sounddevice as sd
import wave
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

# Extract MFCC
features = mfcc(
    audio,
    samplerate=SAMPLE_RATE,
    numcep=N_MFCC
)

# Same processing used during training
features = features.mean(axis=0)

# Load trained prototypes
prototypes = np.load(
    "prototypes.npy",
    allow_pickle=True
).item()

# Calculate distance from each command prototype
distances = {}

for label, prototype in prototypes.items():

    distance = np.linalg.norm(features - prototype)

    distances[label] = distance

# Find closest command
prediction = min(
    distances,
    key=distances.get
)

print("\n-----------------------------")
print("PREDICTED COMMAND:")
print(COMMANDS[prediction])
print("-----------------------------")

print("\nDistances:")

for label, distance in distances.items():

    print(
        f"{COMMANDS[label]}: "
        f"{distance:.2f}"
    )