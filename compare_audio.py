
import os
import wave
import numpy as np

def load_audio(path):
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        audio = np.frombuffer(
            wav.readframes(wav.getnframes()),
            dtype=np.int16
        ).astype(np.float32)

    if rate != 16000 or channels != 1:
        raise ValueError(f"{path}: expected 16000 Hz mono WAV")

    audio /= max(np.max(np.abs(audio)), 1.0)
    return audio

iphone_path = os.path.join(
    "dataset", "LIGHT_ON", "light on 1.wav"
)

test_path = os.path.join(
    "dataset", "LIGHT_ON", "LIGHT_ON_test.wav"
)

a = load_audio(iphone_path)
b = load_audio(test_path)

print("iPhone recording duration:", round(len(a) / 16000, 2), "seconds")
print("Test WAV duration:", round(len(b) / 16000, 2), "seconds")
print("iPhone peak amplitude:", round(float(np.max(np.abs(a))), 3))
print("Test peak amplitude:", round(float(np.max(np.abs(b))), 3))
print("\nAudio comparison complete.")
