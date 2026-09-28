import os
import numpy as np
import librosa

DATASET = r"C:\Users\lokes\Desktop\dataset\vikram_ml\train\lokesh"
REAL_WORLD = r"C:\Users\lokes\Desktop\dataset\real_world_test"

SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 40


def extract_features(path):

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

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel.astype(np.float32)


def collect_features(folder, prefix=None):

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
        and (prefix is None or f.startswith(prefix))
    ])

    features = []

    for filename in files:

        path = os.path.join(
            folder,
            filename
        )

        features.append(
            extract_features(path)
        )

    return np.array(
        features,
        dtype=np.float32
    ), files


print("============================================")
print(" MEL FEATURE COMPARISON")
print("============================================")


# Original Lokesh
original, original_files = collect_features(
    DATASET
)

# Fresh real-world Lokesh
real, real_files = collect_features(
    REAL_WORLD,
    prefix="real_vikram_"
)


print()
print("Original Lokesh feature shape:", original.shape)
print("Real-world feature shape     :", real.shape)


print()
print("--------------------------------------------")
print("OVERALL FEATURE STATISTICS")
print("--------------------------------------------")

print(
    f"Original mean : {np.mean(original):.4f}"
)

print(
    f"Original std  : {np.std(original):.4f}"
)

print(
    f"Original min  : {np.min(original):.4f}"
)

print(
    f"Original max  : {np.max(original):.4f}"
)

print()

print(
    f"Real mean     : {np.mean(real):.4f}"
)

print(
    f"Real std      : {np.std(real):.4f}"
)

print(
    f"Real min      : {np.min(real):.4f}"
)

print(
    f"Real max      : {np.max(real):.4f}"
)


print()
print("--------------------------------------------")
print("PER-FILE FEATURE STATISTICS")
print("--------------------------------------------")

print()
print("ORIGINAL LOKESH")

for filename, feature in zip(
    original_files,
    original
):

    print(
        f"{filename:<25} "
        f"mean={np.mean(feature):.4f} "
        f"std={np.std(feature):.4f}"
    )


print()
print("REAL-WORLD LOKESH")

for filename, feature in zip(
    real_files,
    real
):

    print(
        f"{filename:<25} "
        f"mean={np.mean(feature):.4f} "
        f"std={np.std(feature):.4f}"
    )


print()
print("--------------------------------------------")
print("AVERAGE FEATURE DISTANCE")
print("--------------------------------------------")

original_mean = np.mean(original, axis=0)
real_mean = np.mean(real, axis=0)

distance = np.mean(
    np.abs(original_mean - real_mean)
)

print(
    f"Mean absolute feature difference: "
    f"{distance:.4f}"
)

print()
print("DONE")