import os
import numpy as np
import librosa

BASE = r"C:\Users\lokes\Desktop\dataset\real_world_test"
OUTPUT = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_real_world.npz"

SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 40
MAX_FRAMES = 151

X = []
y = []
names = []

print("========================================")
print(" REAL-WORLD FEATURE EXTRACTION")
print("========================================")
print()

files = sorted(
    f for f in os.listdir(BASE)
    if f.lower().endswith(".wav")
)


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
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=512,
        hop_length=160
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel.astype(np.float32)


for filename in files:

    filepath = os.path.join(BASE, filename)

    features = extract_features(filepath)

    if features.shape[1] < MAX_FRAMES:

        pad_width = MAX_FRAMES - features.shape[1]

        features = np.pad(
            features,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        features = features[:, :MAX_FRAMES]

    X.append(features)

    if filename.startswith("real_vikram_"):
        label = 1
        label_name = "VIKRAM"
    else:
        label = 0
        label_name = "NEGATIVE"

    y.append(label)
    names.append(filename)

    print(f"{filename:<25} -> {label_name}")


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
names = np.array(names)

print()
print("========================================")
print("EXTRACTION COMPLETE")
print("========================================")
print("Samples:", len(X))
print("Feature shape:", X.shape)
print("VIKRAM:", np.sum(y == 1))
print("Negative:", np.sum(y == 0))

np.savez(
    OUTPUT,
    X=X,
    y=y,
    names=names
)

print()
print("Saved:")
print(OUTPUT)
print()