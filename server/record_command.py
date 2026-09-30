import sounddevice as sd
from scipy.io.wavfile import write
import os

SAMPLE_RATE = 16000
DURATION = 3
OUTPUT = r"C:\Users\lokes\Desktop\command_test.wav"

print("Get ready...")
input("Press ENTER, then immediately say: turn on the light")
print("Recording...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)
sd.wait()

write(OUTPUT, SAMPLE_RATE, audio)

print("Recording saved:")
print(OUTPUT)
