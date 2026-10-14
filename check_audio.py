
import os
import wave

folder = os.path.join("dataset", "LIGHT_ON")

files = [
    f for f in os.listdir(folder)
    if f.lower().endswith(".wav")
    and f.lower() != "light_on_test.wav"
]

print("Checking LIGHT_ON recordings...\n")

for filename in sorted(files):
    path = os.path.join(folder, filename)

    try:
        with wave.open(path, "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            duration = audio.getnframes() / sample_rate

        print(
            f"{filename}: "
            f"{sample_rate} Hz, "
            f"{channels} channel(s), "
            f"{duration:.2f} seconds"
        )

    except Exception as error:
        print(f"ERROR: {filename}: {error}")

print("\nTotal recordings:", len(files))
