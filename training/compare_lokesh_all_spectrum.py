import os
import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 1.5

ORIGINAL_DIR = r"C:\Users\lokes\Desktop\dataset\vikram_ml\train\lokesh"
REAL_WORLD_DIR = r"C:\Users\lokes\Desktop\dataset\real_world_test"


def analyze_audio(path):
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

    stft = np.abs(
        librosa.stft(
            audio,
            n_fft=512,
            hop_length=160
        )
    ) ** 2

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=512
    )

    low_mask = (frequencies >= 0) & (frequencies < 300)
    mid_mask = (frequencies >= 300) & (frequencies < 1000)
    high_mask = (frequencies >= 1000) & (frequencies < 4000)

    low_energy = np.sum(stft[low_mask])
    mid_energy = np.sum(stft[mid_mask])
    high_energy = np.sum(stft[high_mask])

    total_energy = (
        low_energy +
        mid_energy +
        high_energy
    )

    low_percent = (low_energy / total_energy) * 100
    mid_percent = (mid_energy / total_energy) * 100
    high_percent = (high_energy / total_energy) * 100

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=512,
            hop_length=160
        )
    )

    return (
        low_percent,
        mid_percent,
        high_percent,
        centroid
    )


def analyze_folder(folder, keyword):
    results = []

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".wav") and keyword in filename.lower():
            path = os.path.join(folder, filename)
            results.append(analyze_audio(path))

    return np.array(results)


print("============================================")
print(" ALL LOKESH SPECTRUM COMPARISON")
print("============================================")


original = analyze_folder(
    ORIGINAL_DIR,
    "vikram_lokesh"
)

real_world = analyze_folder(
    REAL_WORLD_DIR,
    "real_vikram"
)


original_mean = np.mean(original, axis=0)
real_mean = np.mean(real_world, axis=0)


print()
print(f"Original recordings   : {len(original)}")
print(f"Real-world recordings : {len(real_world)}")

print()
print("--------------------------------------------")
print("AVERAGE FREQUENCY DISTRIBUTION")
print("--------------------------------------------")

print()
print("ORIGINAL LOKESH")
print(f"Low energy  : {original_mean[0]:.2f}%")
print(f"Mid energy  : {original_mean[1]:.2f}%")
print(f"High energy : {original_mean[2]:.2f}%")
print(f"Centroid    : {original_mean[3]:.2f} Hz")

print()
print("REAL-WORLD LOKESH")
print(f"Low energy  : {real_mean[0]:.2f}%")
print(f"Mid energy  : {real_mean[1]:.2f}%")
print(f"High energy : {real_mean[2]:.2f}%")
print(f"Centroid    : {real_mean[3]:.2f} Hz")

print()
print("--------------------------------------------")
print("AVERAGE DIFFERENCE")
print("--------------------------------------------")

print(
    f"Low difference  : "
    f"{real_mean[0] - original_mean[0]:+.2f} percentage points"
)

print(
    f"Mid difference  : "
    f"{real_mean[1] - original_mean[1]:+.2f} percentage points"
)

print(
    f"High difference : "
    f"{real_mean[2] - original_mean[2]:+.2f} percentage points"
)

print(
    f"Centroid difference : "
    f"{real_mean[3] - original_mean[3]:+.2f} Hz"
)

print()
print("============================================")
print("DONE")
print("============================================")