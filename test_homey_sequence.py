
import wave
import joblib
import numpy as np
import sounddevice as sd
from python_speech_features import mfcc

RATE = 16000
DURATION = 3
FRAMES = 100
MODEL_PATH = "models/homey_sequence.joblib"

def extract_features(signal):
    signal = signal.astype(np.float32) / 32768.0

    m = mfcc(
        signal,
        samplerate=RATE,
        winlen=0.025,
        winstep=0.010,
        numcep=13,
        nfilt=26,
        nfft=512,
        appendEnergy=True
    )

    old_positions = np.linspace(0, 1, len(m))
    new_positions = np.linspace(0, 1, FRAMES)

    sequence = np.stack([
        np.interp(new_positions, old_positions, m[:, i])
        for i in range(13)
    ], axis=1)

    return sequence.flatten().astype(np.float32).reshape(1, -1)

model = joblib.load(MODEL_PATH)

input("Press Enter, then say 'Turn on the light'...")
print("Recording for 3 seconds. Speak now!")

recording = sd.rec(
    int(RATE * DURATION),
    samplerate=RATE,
    channels=1,
    dtype="int16"
)
sd.wait()

prediction = model.predict(extract_features(recording.flatten()))[0]
print("\nHOMEY sequence prediction:", prediction)
