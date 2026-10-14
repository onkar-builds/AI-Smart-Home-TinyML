
import sounddevice as sd
import wave
import numpy as np

RATE = 16000
SECONDS = 3
OUTPUT = "homey_mic_test.wav"

input("Press Enter, then say: Turn on the light...")
print("Recording now. Speak clearly!")

audio = sd.rec(
    int(RATE * SECONDS),
    samplerate=RATE,
    channels=1,
    dtype="int16"
)
sd.wait()

with wave.open(OUTPUT, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(RATE)
    f.writeframes(audio.tobytes())

signal = audio.astype(np.float32) / 32768.0

print("\nRecording saved:", OUTPUT)
print("Peak level:", round(float(np.max(np.abs(signal))), 4))
print("Average level:", round(float(np.sqrt(np.mean(signal ** 2))), 4))
