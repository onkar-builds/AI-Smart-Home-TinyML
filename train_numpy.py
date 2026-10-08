import numpy as np

# Load preprocessed data
X = np.load("X.npy")
y = np.load("y.npy")

COMMANDS = [
    "LIGHT_ON",
    "LIGHT_OFF",
    "FAN_ON",
    "FAN_OFF"
]

# Convert each audio sample from (299, 13) to one feature vector
X = X.mean(axis=1)

print("Training data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Calculate one prototype (average feature vector) for each command
prototypes = {}

for label in range(len(COMMANDS)):

    samples = X[y == label]

    if len(samples) == 0:
        continue

    prototypes[label] = np.mean(samples, axis=0)

    print(
        f"{COMMANDS[label]}: "
        f"{len(samples)} samples"
    )

# Save prototypes
np.save("prototypes.npy", prototypes, allow_pickle=True)

print("\nTraining complete!")
print("Saved: prototypes.npy")