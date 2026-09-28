import os
import numpy as np
import librosa

INPUT_FOLDER = r"C:\Users\lokes\Desktop\dataset\controlled_test"
OUTPUT_FILE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_controlled.npz"

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


X = []
y = []
names = []

print("========================================")
print(" CONTROLLED FEATURE EXTRACTION")
print("========================================")
print()

files = sorted(
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(".wav")
)

for filename in files:

    file_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    feature = extract_features(file_path)

    X.append(feature)
    y.append(1)
    names.append(filename)

    print(f"{filename:<30} -> VIKRAM")


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
names = np.array(names)

print()
print("========================================")
print(" EXTRACTION COMPLETE")
print("========================================")
print(f"Samples: {len(X)}")
print(f"Feature shape: {X.shape}")
print(f"VIKRAM: {np.sum(y == 1)}")
print()

np.savez(
    OUTPUT_FILE,
    X=X,
    y=y,
    names=names
)

print("Saved:")
print(OUTPUT_FILE)