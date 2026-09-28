import sounddevice as sd
import numpy as np
import wave
import time
import os

# Separate folder for the controlled experiment
OUTPUT_FOLDER = r"C:\Users\lokes\Desktop\dataset\controlled_test"

SAMPLE_RATE = 16000
DURATION = 1.5
CHANNELS = 1

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("========================================")
print(" CONTROLLED VIKRAM RECORDING TEST")
print("========================================")
print()
print("Recording format:")
print("16 kHz | Mono | 16-bit PCM | 1.5 seconds")
print()
print("IMPORTANT:")
print("Use the SAME microphone and same room/environment")
print("that you normally use for your original dataset.")
print()
print("For every recording:")
print("- Sit/stand in the same position")
print("- Keep the microphone at the same distance")
print("- Say VIKRAM naturally")
print("- Do not intentionally speak louder or softer")
print()

num_samples = int(
    input("How many controlled VIKRAM samples? (recommended: 10): ")
)

print()
print(f"Preparing {num_samples} controlled recordings...")
print()

for sample_idx in range(1, num_samples + 1):

    print("----------------------------------------")
    print(f"VIKRAM controlled sample #{sample_idx} of {num_samples}")
    print("----------------------------------------")

    input("Press ENTER when you are ready...")

    for countdown in range(3, 0, -1):
        print(f"Starting in {countdown}...")
        time.sleep(1)

    print("RECORDING NOW... Say 'VIKRAM'!")

    audio_data = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )

    sd.wait()

    print("Recording completed.")

    file_path = os.path.join(
        OUTPUT_FOLDER,
        f"controlled_vikram_{sample_idx}.wav"
    )

    with wave.open(file_path, "wb") as wav_output:
        wav_output.setnchannels(CHANNELS)
        wav_output.setsampwidth(2)
        wav_output.setframerate(SAMPLE_RATE)
        wav_output.writeframes(audio_data.tobytes())

    print(f"Saved: {file_path}")
    print()

print("========================================")
print(" CONTROLLED RECORDING COMPLETE")
print("========================================")
print(f"Files saved in:")
print(OUTPUT_FOLDER)