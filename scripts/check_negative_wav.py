import os
import wave

NEGATIVE_DIR = r"C:\Users\lokes\Desktop\dataset\negative"

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
EXPECTED_FRAMES = 24000

print("Checking negative WAV files...")
print()

total = 0
invalid = 0

for filename in sorted(os.listdir(NEGATIVE_DIR)):
    if not filename.lower().endswith(".wav"):
        continue

    total += 1
    filepath = os.path.join(NEGATIVE_DIR, filename)

    valid = True

    try:
        with wave.open(filepath, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()

            if channels != CHANNELS:
                valid = False

            if sample_width != SAMPLE_WIDTH:
                valid = False

            if sample_rate != SAMPLE_RATE:
                valid = False

            if frames != EXPECTED_FRAMES:
                valid = False

    except Exception:
        valid = False

    print(f"{filename:<25} ValidWAV={valid}")

    if not valid:
        invalid += 1

print()
print("====================================")
print(f"Total negative WAV files: {total}")
print(f"Invalid files: {invalid}")
print("====================================")