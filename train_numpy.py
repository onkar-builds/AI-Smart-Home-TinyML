import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

COMMANDS = [
    "LIGHT_ON",
    "LIGHT_OFF",
    "FAN_ON",
    "FAN_OFF"
]

# Instead of averaging all MFCC frames,
# keep more information from the voice.
features = np.concatenate([
    X.mean(axis=1),
    X.std(axis=1),
    X.min(axis=1),
    X.max(axis=1)
], axis=1)

print("Feature shape:", features.shape)

# Create one reference vector for each command
prototypes = {}

for label in range(len(COMMANDS)):

    samples = features[y == label]

    if len(samples) == 0:
        continue

    prototypes[label] = samples.mean(axis=0)

    print(
        f"{COMMANDS[label]}: "
        f"{len(samples)} samples"
    )

np.save(
    "prototypes.npy",
    prototypes,
    allow_pickle=True
)

print("\nNew model trained successfully!")
print("Saved: prototypes.npy")