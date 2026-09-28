import os
import numpy as np
import tensorflow as tf
import librosa

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"
DATASET = r"C:\Users\lokes\Desktop\dataset\vikram_ml"

MODEL_PATH = os.path.join(BASE, "vikram_cnn.keras")

SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 40

SPEAKERS = [
    ("dtss", "train"),
    ("lokesh", "train"),
    ("s.akshitha", "train"),
    ("lavanya", "validation"),
    ("mohana", "test")
]


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


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print()
print("============================================")
print(" SPEAKER-WISE VIKRAM EVALUATION")
print("============================================")

for speaker, split in SPEAKERS:

    folder = os.path.join(
        DATASET,
        split,
        speaker
    )

    if not os.path.exists(folder):
        print()
        print("Folder not found:", folder)
        continue

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    X = []

    for filename in files:

        path = os.path.join(folder, filename)

        try:
            features = extract_features(path)
            X.append(features)

        except Exception as e:
            print("ERROR:", filename, "->", e)

    if not X:
        continue

    X = np.array(
        X,
        dtype=np.float32
    )

    X = X[..., np.newaxis]

    probabilities = model.predict(
        X,
        verbose=0
    ).flatten()

    predictions = probabilities >= 0.5

    correct = np.sum(predictions)

    accuracy = correct / len(files)

    print()
    print("--------------------------------------------")
    print(f"Speaker : {speaker}")
    print(f"Split   : {split}")
    print(f"Samples : {len(files)}")
    print(f"Detected as VIKRAM: {correct}/{len(files)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
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
print("============================================")
print(" COMPLETE")
print("============================================")