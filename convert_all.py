
import os
import subprocess

FFMPEG = (
    r"C:\Users\conta\Downloads\ffmpeg-2026-10-12-git-615b07170e-essentials_build"
    r"\ffmpeg-2026-10-12-git-615b07170e-essentials_build\bin\ffmpeg.exe"
)

FOLDER = os.path.join("dataset", "LIGHT_ON")

converted = 0
failed = 0

for filename in os.listdir(FOLDER):
    if not filename.lower().endswith(".mp4"):
        continue

    source = os.path.join(FOLDER, filename)
    output_name = os.path.splitext(filename)[0] + ".wav"
    output = os.path.join(FOLDER, output_name)

    result = subprocess.run(
        [
            FFMPEG, "-y", "-i", source,
            "-vn", "-ac", "1", "-ar", "16000",
            "-sample_fmt", "s16", output
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        converted += 1
        print("Converted:", filename, "->", output_name)
    else:
        failed += 1
        print("FAILED:", filename)
        print(result.stderr[-500:])

print("\nConversion finished!")
print("Successful:", converted)
print("Failed:", failed)
