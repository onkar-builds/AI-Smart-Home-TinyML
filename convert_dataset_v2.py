
import os
import subprocess
from glob import glob

FFMPEG = r"C:\Users\conta\Downloads\ffmpeg-2026-10-12-git-615b07170e-essentials_build\ffmpeg-2026-10-12-git-615b07170e-essentials_build\bin\ffmpeg.exe"

folders = ["LIGHT_ON", "LIGHT_OFF", "FAN_ON", "FAN_OFF"]

for folder in folders:
    input_dir = os.path.join("dataset", folder)
    output_dir = os.path.join("dataset_v2", folder)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, "*.mp4")))

    for index, source in enumerate(files, start=1):
        output = os.path.join(output_dir, f"{folder}_{index:02d}.wav")

        command = [
            FFMPEG, "-y", "-i", source,
            "-vn", "-ac", "1", "-ar", "16000",
            "-sample_fmt", "s16", output
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            print("Converted:", folder, "->", os.path.basename(output))
        else:
            print("FAILED:", source)
            print(result.stderr[-1000:])

print("\nConversion complete.")
