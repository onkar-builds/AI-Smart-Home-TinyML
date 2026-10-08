import os
import wave
import numpy as np
from python_speech_features import mfcc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

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

    # Convert 2D MFCC into 1D
    return features.flatten()


X = []
y = []

print("Loading dataset...\n")

for label, command in enumerate(COMMANDS):

    folder = os.path.join(DATASET_PATH, command)

    if not os.path.exists(folder):
        print(f"Skipping {command}")
        continue

    for filename in sorted(os.listdir(folder)):

        if filename.endswith(".wav"):

            file_path = os.path.join(folder, filename)

            features = extract_features(file_path)

            X.append(features)
            y.append(label)

            print(f"Loaded: {file_path}")


X = np.array(X)
y = np.array(y)

print("\nDataset loaded!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Total samples:", len(X))


# Check minimum dataset size
if len(X) < 20:

    print("\nNot enough data for training yet.")

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining model...")

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining complete!")
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=COMMANDS,
            zero_division=0
        )
    )

    joblib.dump(model, "tinyml_voice_model.pkl")

    print("\nModel saved as: tinyml_voice_model.pkl")