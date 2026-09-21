import os
import shutil
import random

SOURCE = r"C:\Users\lokes\Desktop\dataset\vikram"
DEST = r"C:\Users\lokes\Desktop\dataset\vikram_ml"

random.seed(42)

# Get speaker folders
speakers = [
    folder
    for folder in os.listdir(SOURCE)
    if os.path.isdir(os.path.join(SOURCE, folder))
]

print("Speakers found:")
for speaker in sorted(speakers):
    print(" -", speaker)

print()

# We have 4 speakers.
# Keep speakers separate between train, validation and test.
#
# Train:      dtss + lokesh
# Validation: lavanya
# Test:       mohana

speaker_split = {
    "train": ["dtss", "lokesh"],
    "validation": ["lavanya"],
    "test": ["mohana"]
}

# Copy files
for split, split_speakers in speaker_split.items():
    split_folder = os.path.join(DEST, split)

    for speaker in split_speakers:
        source_folder = os.path.join(SOURCE, speaker)

        if not os.path.exists(source_folder):
            print("WARNING: speaker folder not found:", speaker)
            continue

        files = [
            f for f in os.listdir(source_folder)
            if f.lower().endswith(".wav")
        ]

        random.shuffle(files)

        speaker_dest = os.path.join(split_folder, speaker)
        os.makedirs(speaker_dest, exist_ok=True)

        for filename in files:
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(speaker_dest, filename)

            shutil.copy2(source_file, destination_file)

        print(f"{speaker}: {len(files)} files -> {split}")

print()
print("====================================")
print("Dataset split completed.")
print("====================================")