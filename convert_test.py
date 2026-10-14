
import os
import subprocess

FFMPEG_BIN = (
    r"C:\Users\conta\Downloads\ffmpeg-2026-10-12-git-615b07170e-essentials_build"
    r"\ffmpeg-2026-10-12-git-615b07170e-essentials_build\bin"
)

ffmpeg = os.path.join(FFMPEG_BIN, "ffmpeg.exe")

source = os.path.join("dataset", "LIGHT_ON", "light on 1.mp4")
output = os.path.join("dataset", "LIGHT_ON", "LIGHT_ON_test.wav")

if not os.path.isfile(ffmpeg):
    print("ERROR: ffmpeg.exe not found")
elif not os.path.isfile(source):
    print("ERROR: Source recording not found:", source)
else:
    result = subprocess.run([
        ffmpeg,
        "-y",
        "-i", source,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("Conversion successful!")
        print("Saved:", output)
    else:
        print("Conversion failed:")
        print(result.stderr[-3000:])
