import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
DURATION = 1.5

print("Microphone RMS test")
input("Press ENTER, then say VIKRAM clearly...")

print("\nRecording for 1.5 seconds...")

audio = sd.rec(
    int(SAMPLE_RATE * DURATION),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio.flatten()

rms = float(np.sqrt(np.mean(audio ** 2)))
maximum = float(np.max(np.abs(audio)))

print("\n------------------------------")
print(f"RMS: {rms:.8f}")
print(f"Max: {maximum:.8f}")
print("------------------------------")