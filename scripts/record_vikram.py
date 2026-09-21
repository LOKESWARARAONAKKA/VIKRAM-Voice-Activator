import sounddevice as sd
import numpy as np
import wave
import time
import os

# Ask for the person's name to organize files automatically
user_name = input("👤 Enter the name of the team member recording right now: ").strip().lower()
if not user_name:
    user_name = "unknown_user"

# Create a clean folder structure for this user
output_folder = f"dataset/vikram/{user_name}"
os.makedirs(output_folder, exist_ok=True)

SAMPLE_RATE = 16000  # 16kHz matching the ISRO hardware constraints
DURATION = 1.5       # 1.5-second time window for each recording

print("\n🎙️ ISRO DATASET CREATOR CORE INITIALIZED.")
num_samples = int(input("How many samples do you want to record in this session? (e.g., 20): "))

print(f"\nPerfect. Preparing to record {num_samples} samples for user: '{user_name}'.")
print()

for sample_idx in range(1, num_samples + 1):

    print("----------------------------------------")
    print(f"👉 VIKRAM sample #{sample_idx} of {num_samples}")
    print("----------------------------------------")

    # Wait for ENTER before every individual recording
    input("Press ENTER to start this recording...")

    # 3-second countdown
    for countdown in range(3, 0, -1):
        print(f"Starting in {countdown}...")
        time.sleep(1)

    print("🔴 RECORDING NOW... Say 'VIKRAM'!")

    # Capture the raw 16kHz audio
    audio_data = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    print("⏹️ Recording completed.")

    # Export as WAV
    file_path = f"{output_folder}/vikram_{user_name}_{sample_idx}.wav"

    with wave.open(file_path, 'wb') as wav_output:
        wav_output.setnchannels(1)     # Mono
        wav_output.setsampwidth(2)     # 16-bit PCM
        wav_output.setframerate(SAMPLE_RATE)
        wav_output.writeframes(audio_data.tobytes())

    print(f"💾 File saved: '{file_path}'")
    print()

print(f"🏁 Session complete for {user_name}!")
print(f"Your files are ready in the '{output_folder}' folder.")