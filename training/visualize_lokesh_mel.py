import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 40


def extract_mel(path):

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    expected_length = int(SAMPLE_RATE * DURATION)

    if len(audio) < expected_length:
        audio = np.pad(
            audio,
            (0, expected_length - len(audio))
        )
    else:
        audio = audio[:expected_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=512,
        hop_length=160
    )

    return librosa.power_to_db(
        mel,
        ref=np.max
    )


ORIGINAL = r"C:\Users\lokes\Desktop\dataset\vikram_ml\train\lokesh\vikram_lokesh_10.wav"

REAL_WORLD = r"C:\Users\lokes\Desktop\dataset\real_world_test\real_vikram_1.wav"


original_mel = extract_mel(ORIGINAL)
real_mel = extract_mel(REAL_WORLD)


print("============================================")
print(" MEL SPECTROGRAM COMPARISON")
print("============================================")

print()
print("Original shape:", original_mel.shape)
print("Real-world shape:", real_mel.shape)

print()
print("Original mean:", np.mean(original_mel))
print("Real-world mean:", np.mean(real_mel))

print()
print("Mean absolute difference:")

difference = np.mean(
    np.abs(original_mel - real_mel)
)

print(difference)

print()
print("Opening comparison plots...")


plt.figure(figsize=(10, 4))

plt.imshow(
    original_mel,
    aspect="auto",
    origin="lower"
)

plt.title("Original Lokesh - vikram_lokesh_10.wav")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.colorbar()

plt.tight_layout()


plt.figure(figsize=(10, 4))

plt.imshow(
    real_mel,
    aspect="auto",
    origin="lower"
)

plt.title("Real-world Lokesh - real_vikram_1.wav")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.colorbar()

plt.tight_layout()

plt.show()

print()
print("DONE")