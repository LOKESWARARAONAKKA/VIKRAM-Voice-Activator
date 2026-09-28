import os
import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

MODEL_PATH = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\vikram_cnn_adapted.keras"

SAMPLE_RATE = 16000
DURATION = 1.5
N_MELS = 40

# Start with 0.50 for the demo.
THRESHOLD = 0.50


def extract_features(audio):
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


print("========================================")
print(" VIKRAM - LIVE VOICE ACTIVATOR")
print("========================================")
print()

print("Loading adapted model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print()
print("Wake word : VIKRAM")
print(f"Threshold : {THRESHOLD:.2f}")
print()
print("The system will listen for 1.5 seconds at a time.")
print("Say VIKRAM clearly when prompted.")
print()
print("Press Ctrl+C to stop.")
print()

try:

    while True:

        input("Press ENTER, then say VIKRAM...")

        print("Listening...")

        recording = sd.rec(
    int(SAMPLE_RATE * DURATION),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=1
)

        sd.wait()

        audio = recording[:, 0]

        features = extract_features(audio)

        X = features[np.newaxis, ..., np.newaxis]

        probability = float(
            model.predict(X, verbose=0)[0][0]
        )

        if probability >= THRESHOLD:

            print()
            print(">>> VIKRAM DETECTED <<<")
            print(
                f"Confidence: {probability:.4f}"
            )

        else:

            print()
            print(
                f"No wake word detected "
                f"(confidence={probability:.4f})"
            )

        print()

except KeyboardInterrupt:

    print()
    print("========================================")
    print(" Demo stopped.")
    print("========================================")