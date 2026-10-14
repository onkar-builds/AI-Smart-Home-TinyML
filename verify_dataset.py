
import os
import wave
from glob import glob

folders = ["LIGHT_ON", "LIGHT_OFF", "FAN_ON", "FAN_OFF"]
total = 0
all_valid = True

for folder in folders:
    files = sorted(glob(os.path.join("dataset_v2", folder, "*.wav")))
    print(f"\n{folder}: {len(files)} WAV files")
    
    for path in files:
        try:
            with wave.open(path, "rb") as audio:
                channels = audio.getnchannels()
                rate = audio.getframerate()
                width = audio.getsampwidth()
                frames = audio.getnframes()

                valid = channels == 1 and rate == 16000 and width == 2
                if not valid:
                    all_valid = False
                    print("  CHECK:", os.path.basename(path),
                          channels, rate, width)
        except Exception as error:
            all_valid = False
            print("  ERROR:", os.path.basename(path), error)

    total += len(files)

print(f"\nTotal WAV files: {total}")
print("Audio format check:", "PASSED" if all_valid else "FAILED")
