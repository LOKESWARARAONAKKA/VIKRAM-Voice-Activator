import sounddevice as sd
import numpy as np
import wave
import os

SAVE_DIR = r"C:\Users\lokes\Desktop\dataset\negative"

SAMPLE_RATE = 16000
DURATION = 1.5
CHANNELS = 1

os.makedirs(SAVE_DIR, exist_ok=True)

print("====================================")
print(" VIKRAM NEGATIVE DATASET RECORDER")
print("====================================")
print()
print("Record sounds that are NOT the wake word.")
print()
print("Examples:")
print("- normal speech")
print("- different words")
print("- short conversations")
print("- room/background sounds")
print("- keyboard or desk sounds")
print("- fan/air-conditioner noise")
print()

input("Press ENTER to start...")

num_samples = int(input("How many negative samples? (e.g., 30): "))

print()
print(f"Preparing to record {num_samples} negative samples.")
print()

for sample_idx in range(1, num_samples + 1):

    print("------------------------------------")
    print(f"Negative sample #{sample_idx} of {num_samples}")
    print("------------------------------------")

    input("Press ENTER to start this recording...")

    print()
    print("RECORDING...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.int16
    )

    sd.wait()

    filename = f"negative_{sample_idx}.wav"
    filepath = os.path.join(SAVE_DIR, filename)

    with wave.open(filepath, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.tobytes())

    print()
    print("Saved:", filename)
    print()

print("====================================")
print("Negative recording session complete.")
print("====================================")