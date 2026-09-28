import os
import numpy as np
import tensorflow as tf
import librosa

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"
DATASET = r"C:\Users\lokes\Desktop\dataset\vikram_ml"
REAL_WORLD = r"C:\Users\lokes\Desktop\dataset\real_world_test"

MODEL_PATH = os.path.join(BASE, "vikram_cnn.keras")

SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 40


def extract_features(audio_file):

    audio, sr = librosa.load(
        audio_file,
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


def evaluate_folder(model, folder, label):

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    X = []

    for filename in files:

        path = os.path.join(folder, filename)

        features = extract_features(path)

        X.append(features)

    X = np.array(
        X,
        dtype=np.float32
    )

    X = X[..., np.newaxis]

    probabilities = model.predict(
        X,
        verbose=0
    ).flatten()

    print()
    print("--------------------------------------------")
    print(label)
    print("--------------------------------------------")

    for filename, probability in zip(
        files,
        probabilities
    ):

        print(
            f"{filename:<25} "
            f"Confidence={probability:.4f}"
        )

    print()
    print(
        f"Average confidence: "
        f"{np.mean(probabilities):.4f}"
    )

    print(
        f"Minimum confidence: "
        f"{np.min(probabilities):.4f}"
    )

    print(
        f"Maximum confidence: "
        f"{np.max(probabilities):.4f}"
    )

    return probabilities


print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print("============================================")
print(" LOKESH vs REAL-WORLD COMPARISON")
print("============================================")


# Original Lokesh training recordings
lokesh_folder = os.path.join(
    DATASET,
    "train",
    "lokesh"
)

lokesh_probabilities = evaluate_folder(
    model,
    lokesh_folder,
    "ORIGINAL LOKESH TRAINING RECORDINGS"
)


# Fresh real-world VIKRAM recordings
real_files = sorted([
    f for f in os.listdir(REAL_WORLD)
    if f.lower().startswith("real_vikram_")
    and f.lower().endswith(".wav")
])

X_real = []

for filename in real_files:

    path = os.path.join(
        REAL_WORLD,
        filename
    )

    features = extract_features(path)

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

print()
print("--------------------------------------------")
print("FRESH REAL-WORLD LOKESH RECORDINGS")
print("--------------------------------------------")

for filename, probability in zip(
    real_files,
    real_probabilities
):

    print(
        f"{filename:<25} "
        f"Confidence={probability:.4f}"
    )

print()
print(
    f"Average confidence: "
    f"{np.mean(real_probabilities):.4f}"
)

print(
    f"Minimum confidence: "
    f"{np.min(real_probabilities):.4f}"
)

print(
    f"Maximum confidence: "
    f"{np.max(real_probabilities):.4f}"
)


print()
print("============================================")
print(" FINAL COMPARISON")
print("============================================")

print(
    f"Original Lokesh average : "
    f"{np.mean(lokesh_probabilities):.4f}"
)

print(
    f"Real-world average      : "
    f"{np.mean(real_probabilities):.4f}"
)

print(
    f"Difference              : "
    f"{np.mean(lokesh_probabilities) - np.mean(real_probabilities):.4f}"
)

print("============================================")