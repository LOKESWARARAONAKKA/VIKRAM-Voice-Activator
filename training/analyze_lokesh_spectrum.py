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

    # STFT power spectrum
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
    low = stft[(frequencies >= 0) & (frequencies < 300)]
    mid = stft[(frequencies >= 300) & (frequencies < 1000)]
    high = stft[(frequencies >= 1000) & (frequencies < 4000)]

    low_energy = np.mean(low)
    mid_energy = np.mean(mid)
    high_energy = np.mean(high)

    total_energy = np.mean(stft)

    low_ratio = low_energy / total_energy
    mid_ratio = mid_energy / total_energy
    high_ratio = high_energy / total_energy

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=512,
            hop_length=160
        )
    )

    return (
        low_energy,
        mid_energy,
        high_energy,
        low_ratio,
        mid_ratio,
        high_ratio,
        spectral_centroid
    )


original = analyze_audio(ORIGINAL)
real_world = analyze_audio(REAL_WORLD)


print("============================================")
print(" LOKESH SPECTRUM COMPARISON")
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

print(f"Low energy ratio      : {original[3]:.6f}")
print(f"Mid energy ratio      : {original[4]:.6f}")
print(f"High energy ratio     : {original[5]:.6f}")
print(f"Spectral centroid     : {original[6]:.2f} Hz")

print()
print("--------------------------------------------")
print("REAL-WORLD LOKESH")
print("--------------------------------------------")

print(f"Low energy ratio      : {real_world[3]:.6f}")
print(f"Mid energy ratio      : {real_world[4]:.6f}")
print(f"High energy ratio     : {real_world[5]:.6f}")
print(f"Spectral centroid     : {real_world[6]:.2f} Hz")

print()
print("--------------------------------------------")
print("DIFFERENCE")
print("--------------------------------------------")

print(
    f"Low ratio difference  : "
    f"{real_world[3] - original[3]:.6f}"
)

print(
    f"Mid ratio difference  : "
    f"{real_world[4] - original[4]:.6f}"
)

print(
    f"High ratio difference : "
    f"{real_world[5] - original[5]:.6f}"
)

print(
    f"Centroid difference   : "
    f"{real_world[6] - original[6]:.2f} Hz"
)

print()
print("DONE")