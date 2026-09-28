import os
import numpy as np
import librosa

# ============================================
# VIKRAM Voice Activator
# Feature Extraction
# ============================================

DATASET_DIR = r"C:\Users\lokes\Desktop\dataset\vikram_ml"
OUTPUT_FILE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features.npz"

SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 40

# Speaker folders are positive "VIKRAM" samples
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

    # Make every recording exactly 1.5 seconds
    expected_length = int(SAMPLE_RATE * DURATION)

    if len(audio) < expected_length:
        audio = np.pad(
            audio,
            (0, expected_length - len(audio))
        )
    else:
        audio = audio[:expected_length]

    # Create Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=512,
        hop_length=160
    )

    # Convert to logarithmic scale
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

            print(
                "ERROR:",
                filename,
                "->",
                e
            )


def main():

    X = []
    y = []

    print("============================================")
    print(" VIKRAM FEATURE EXTRACTION")
    print("============================================")
    print()

    for split in ["train", "validation", "test"]:

        split_path = os.path.join(
            DATASET_DIR,
            split
        )

        print(f"Processing {split.upper()} dataset...")

        # ----------------------------------------
        # Positive VIKRAM samples
        # ----------------------------------------

        positive_count_before = len(X)

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
                    f"  {speaker}: {len(files)} files -> VIKRAM"
                )

                process_folder(
                    speaker_path,
                    1,
                    X,
                    y
                )

        positive_count_after = len(X)

        print(
            f"  Positive samples processed: "
            f"{positive_count_after - positive_count_before}"
        )

        # ----------------------------------------
        # Negative samples
        # ----------------------------------------

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
                f"  negative: {len(files)} files -> NEGATIVE"
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

        print()

    # Convert to NumPy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print("============================================")
    print(" FEATURE EXTRACTION COMPLETE")
    print("============================================")

    print("Total samples:", len(X))
    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)

    print()
    print("VIKRAM samples:", np.sum(y == 1))
    print("Negative samples:", np.sum(y == 0))

    # Save features
    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y
    )

    print()
    print("Features saved to:")
    print(OUTPUT_FILE)
    print()


if __name__ == "__main__":
    main()