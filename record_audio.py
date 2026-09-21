import sounddevice as sd
import wave
import os

SAMPLE_RATE = 16000
DURATION = 3
CHANNELS = 1


def record_command(command, number):
    folder = os.path.join("dataset", command)
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"{command}_{number}.wav")

    print(f"\nGet ready to record: {command}")
    input("Press ENTER and speak...")

    print("Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )

    sd.wait()

    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())

    print(f"Saved: {filename}")


command = input(
    "\nEnter command "
    "(LIGHT_ON / LIGHT_OFF / FAN_ON / FAN_OFF): "
).strip().upper()

number = input("Enter recording number: ").strip()

record_command(command, number)

print("\nRecording complete!")