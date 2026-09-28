import os
import numpy as np
import librosa

DATASET_DIR = r"C:\Users\lokes\Desktop\dataset\vikram_ml"
OUTPUT_DIR = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"

SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 40

POSITIVE_SPEAKERS = [
    "dtss",
    "lokesh",
    "s.akshitha",
    "lavanya",
    "mohana"
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


def process_folder(folder_path, label, X, y):

    if not os.path.exists(folder_path):
        print("WARNING: Folder not found:", folder_path)
        return

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".wav")
    ]

    for filename in sorted(files):

        file_path = os.path.join(
            folder_path,
            filename
        )

        try:
            features = extract_features(file_path)

            X.append(features)
            y.append(label)

        except Exception as e:
            print("ERROR:", filename, "->", e)


def process_split(split):

    X = []
    y = []

    split_path = os.path.join(
        DATASET_DIR,
        split
    )

    print()
    print("============================================")
    print("Processing:", split.upper())
    print("============================================")

    # Positive VIKRAM samples
    for speaker in POSITIVE_SPEAKERS:

        speaker_path = os.path.join(
            split_path,
            speaker
        )

        if os.path.exists(speaker_path):

            files = [
                f for f in os.listdir(speaker_path)
                if f.lower().endswith(".wav")
            ]

            print(
                f"{speaker}: {len(files)} files -> VIKRAM"
            )

            process_folder(
                speaker_path,
                1,
                X,
                y
            )

    # Negative samples
    negative_path = os.path.join(
        split_path,
        "negative"
    )

    if os.path.exists(negative_path):

        files = [
            f for f in os.listdir(negative_path)
            if f.lower().endswith(".wav")
        ]

        print(
            f"negative: {len(files)} files -> NEGATIVE"
        )

        process_folder(
            negative_path,
            0,
            X,
            y
        )

    else:

        print(
            "WARNING: Negative folder not found:",
            negative_path
        )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"features_{split}.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    print()
    print("Samples:", len(X))
    print("Feature shape:", X.shape)
    print("VIKRAM:", np.sum(y == 1))
    print("Negative:", np.sum(y == 0))
    print("Saved:", output_file)

    return len(X)


def main():

    print()
    print("============================================")
    print(" VIKRAM SPLIT-AWARE FEATURE EXTRACTION")
    print("============================================")

    train_count = process_split("train")
    validation_count = process_split("validation")
    test_count = process_split("test")

    print()
    print("============================================")
    print(" COMPLETE")
    print("============================================")

    print("Train samples:", train_count)
    print("Validation samples:", validation_count)
    print("Test samples:", test_count)
    print(
        "Total samples:",
        train_count + validation_count + test_count
    )


if __name__ == "__main__":
    main()