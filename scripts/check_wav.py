import os
import wave

root = r"C:\Users\lokes\Desktop\dataset\vikram"

print("Checking WAV files...")
print()

total = 0
invalid = 0

for folder, _, files in os.walk(root):
    for filename in sorted(files):
        if filename.lower().endswith(".wav"):
            total += 1
            path = os.path.join(folder, filename)

            try:
                with wave.open(path, "rb") as wav:
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    sample_rate = wav.getframerate()
                    frames = wav.getnframes()

                    valid = (
                        channels == 1
                        and sample_width == 2
                        and sample_rate == 16000
                        and frames == 24000
                    )

                    if not valid:
                        invalid += 1

                    print(
                        f"{os.path.basename(folder):8} "
                        f"{filename:25} "
                        f"ValidWAV={valid}"
                    )

            except Exception as e:
                invalid += 1
                print(
                    f"{os.path.basename(folder):8} "
                    f"{filename:25} "
                    f"ERROR: {e}"
                )

print()
print("====================================")
print("Total WAV files:", total)
print("Invalid files:", invalid)
print("====================================")