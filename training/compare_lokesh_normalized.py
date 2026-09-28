import os
import numpy as np
import librosa
import tensorflow as tf

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"
MODEL_PATH = os.path.join(BASE, "vikram_cnn.keras")

ORIGINAL = r"C:\Users\lokes\Desktop\dataset\vikram_ml\train\lokesh"
REAL = r"C:\Users\lokes\Desktop\dataset\real_world_test"

SAMPLE_RATE = 16000
DURATION = 1.5
TARGET_RMS = 0.0175


def load_audio(path):
    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    expected = int(SAMPLE_RATE * DURATION)

    if len(audio) < expected:
        audio = np.pad(
            audio,
            (0, expected - len(audio))
        )
    else:
        audio = audio[:expected]

    return audio


def normalize_rms(audio):
    rms = np.sqrt(np.mean(audio ** 2))

    if rms > 0:
        audio = audio * (TARGET_RMS / rms)

    return audio


def extract_features(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=40,
        n_fft=512,
        hop_length=160
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel.astype(np.float32)


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

original_files = sorted([
    f for f in os.listdir(ORIGINAL)
    if f.lower().endswith(".wav")
])

real_files = sorted([
    f for f in os.listdir(REAL)
    if f.startswith("real_vikram_")
    and f.lower().endswith(".wav")
])


print()
print("============================================")
print(" RMS NORMALIZATION EXPERIMENT")
print("============================================")
print()
print(f"Target RMS: {TARGET_RMS:.4f}")


# ------------------------------------------------
# ORIGINAL LOKESH
# ------------------------------------------------

X_original = []

for filename in original_files:

    path = os.path.join(
        ORIGINAL,
        filename
    )

    audio = load_audio(path)

    audio = normalize_rms(audio)

    features = extract_features(audio)

    X_original.append(features)


X_original = np.array(
    X_original,
    dtype=np.float32
)

X_original = X_original[..., np.newaxis]

original_probabilities = model.predict(
    X_original,
    verbose=0
).flatten()


# ------------------------------------------------
# REAL-WORLD
# ------------------------------------------------

X_real = []

for filename in real_files:

    path = os.path.join(
        REAL,
        filename
    )

    audio = load_audio(path)

    original_rms = np.sqrt(
        np.mean(audio ** 2)
    )

    audio = normalize_rms(audio)

    features = extract_features(audio)

    X_real.append(features)


X_real = np.array(
    X_real,
    dtype=np.float32
)

X_real = X_real[..., np.newaxis]

real_probabilities = model.predict(
    X_real,
    verbose=0
).flatten()


# ------------------------------------------------
# RESULTS
# ------------------------------------------------

print()
print("--------------------------------------------")
print("NORMALIZED ORIGINAL LOKESH")
print("--------------------------------------------")

for filename, probability in zip(
    original_files,
    original_probabilities
):

    print(
        f"{filename:<25} "
        f"{probability:.4f}"
    )


print()
print("--------------------------------------------")
print("NORMALIZED REAL-WORLD VIKRAM")
print("--------------------------------------------")

for filename, probability in zip(
    real_files,
    real_probabilities
):

    print(
        f"{filename:<25} "
        f"{probability:.4f}"
    )


print()
print("============================================")
print(" SUMMARY")
print("============================================")

print(
    f"Original Lokesh average : "
    f"{np.mean(original_probabilities):.4f}"
)

print(
    f"Real-world average      : "
    f"{np.mean(real_probabilities):.4f}"
)

print(
    f"Difference              : "
    f"{np.mean(original_probabilities) - np.mean(real_probabilities):.4f}"
)

print()
print("DONE")