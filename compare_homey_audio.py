
import wave
import numpy as np
from python_speech_features import mfcc

def extract(path):
    with wave.open(path, "rb") as audio:
        rate = audio.getframerate()
        signal = np.frombuffer(
            audio.readframes(audio.getnframes()),
            dtype=np.int16
        ).astype(np.float32) / 32768.0

    m = mfcc(
        signal,
        samplerate=rate,
        winlen=0.025,
        winstep=0.010,
        numcep=13,
        nfilt=26,
        nfft=512,
        appendEnergy=True
    )
    return np.concatenate([m.mean(axis=0), m.std(axis=0)])

training = extract("dataset_v2/LIGHT_ON/LIGHT_ON_01.wav")
microphone = extract("homey_mic_test.wav")

print("Training feature shape:", training.shape)
print("Microphone feature shape:", microphone.shape)
print("Mean absolute feature difference:",
      round(float(np.mean(np.abs(training - microphone))), 4))
print("Training feature mean:", round(float(training.mean()), 4))
print("Microphone feature mean:", round(float(microphone.mean()), 4))
