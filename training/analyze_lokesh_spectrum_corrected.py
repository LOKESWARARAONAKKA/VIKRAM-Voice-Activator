import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 1.5

ORIGINAL = r"C:\Users\lokes\Desktop\dataset\vikram_ml\train\lokesh\vikram_lokesh_10.wav"
REAL_WORLD = r"C:\Users\lokes\Desktop\dataset\real_world_test\real_vikram_1.wav"


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

    # Frequency bands
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

    spectral_centroid = np.mean(
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
        spectral_centroid
    )


original = analyze_audio(ORIGINAL)
real_world = analyze_audio(REAL_WORLD)


print("============================================")
print(" CORRECTED LOKESH SPECTRUM COMPARISON")
print("============================================")

print()
print("Frequency bands:")
print("Low  : 0 - 300 Hz")
print("Mid  : 300 - 1000 Hz")
print("High : 1000 - 4000 Hz")

print()
print("--------------------------------------------")
print("ORIGINAL LOKESH")
print("--------------------------------------------")

print(f"Low energy  : {original[0]:.2f}%")
print(f"Mid energy  : {original[1]:.2f}%")
print(f"High energy : {original[2]:.2f}%")
print(f"Centroid    : {original[3]:.2f} Hz")

print()
print("--------------------------------------------")
print("REAL-WORLD LOKESH")
print("--------------------------------------------")

print(f"Low energy  : {real_world[0]:.2f}%")
print(f"Mid energy  : {real_world[1]:.2f}%")
print(f"High energy : {real_world[2]:.2f}%")
print(f"Centroid    : {real_world[3]:.2f} Hz")

print()
print("--------------------------------------------")
print("DIFFERENCE")
print("--------------------------------------------")

print(
    f"Low difference  : "
    f"{real_world[0] - original[0]:+.2f} percentage points"
)

print(
    f"Mid difference  : "
    f"{real_world[1] - original[1]:+.2f} percentage points"
)

print(
    f"High difference : "
    f"{real_world[2] - original[2]:+.2f} percentage points"
)

print(
    f"Centroid difference : "
    f"{real_world[3] - original[3]:+.2f} Hz"
)

print()
print("DONE")