import os
import numpy as np
import tensorflow as tf

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"

MODEL_PATH = os.path.join(
    BASE,
    "vikram_cnn_adapted.keras"
)

FEATURES_PATH = os.path.join(
    BASE,
    "features_real_world.npz"
)

print("========================================")
print(" ADAPTED MODEL - REAL WORLD TEST")
print("========================================")

print("\nLoading adapted model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading real-world features...")
data = np.load(FEATURES_PATH)

X = data["X"]
y = data["y"]
names = data["names"]

X = X[..., np.newaxis]

print()
print("Samples:", len(X))
print("Feature shape:", X.shape)

probabilities = model.predict(
    X,
    verbose=0
).flatten()

threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(int)

print()
print("========================================")
print(" INDIVIDUAL RESULTS")
print("========================================")
print()

correct = 0

for i in range(len(X)):

    actual_name = (
        "VIKRAM"
        if y[i] == 1
        else "NEGATIVE"
    )

    predicted_name = (
        "VIKRAM"
        if predictions[i] == 1
        else "NEGATIVE"
    )

    status = (
        "CORRECT"
        if predictions[i] == y[i]
        else "WRONG"
    )

    if predictions[i] == y[i]:
        correct += 1

    print(
        f"{i+1:02d}. "
        f"{names[i]:<25} "
        f"Actual={actual_name:<8} "
        f"Predicted={predicted_name:<8} "
        f"Confidence={probabilities[i]:.4f} "
        f"[{status}]"
    )

print()
print("========================================")
print(" REAL-WORLD RESULT")
print("========================================")
print()

print(f"Threshold          : {threshold:.2f}")
print(f"Correct predictions : {correct} / {len(X)}")
print(f"Overall accuracy    : {correct / len(X) * 100:.2f}%")

print()

# Separate VIKRAM detection rate
positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]

positive_correct = np.sum(
    predictions[positive_indices] == 1
)

negative_correct = np.sum(
    predictions[negative_indices] == 0
)

print(
    f"VIKRAM detection    : "
    f"{positive_correct} / {len(positive_indices)} "
    f"({positive_correct / len(positive_indices) * 100:.2f}%)"
)

print(
    f"Negative rejection  : "
    f"{negative_correct} / {len(negative_indices)} "
    f"({negative_correct / len(negative_indices) * 100:.2f}%)"
)

print()
print("DONE")