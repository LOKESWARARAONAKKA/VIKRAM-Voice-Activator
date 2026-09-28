import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"

MODEL_PATH = os.path.join(
    BASE,
    "vikram_cnn_adapted.keras"
)

FEATURES_PATH = os.path.join(
    BASE,
    "features_controlled.npz"
)

print("========================================")
print(" ADAPTED VIKRAM MODEL TEST")
print("========================================")

print("\nLoading adapted model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading controlled features...")
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

for i in range(len(X)):

    predicted = (
        "VIKRAM"
        if predictions[i] == 1
        else "NEGATIVE"
    )

    status = (
        "CORRECT"
        if predictions[i] == y[i]
        else "WRONG"
    )

    print(
        f"{i+1:02d}. "
        f"{names[i]:<30} "
        f"Confidence={probabilities[i]:.4f} "
        f"Predicted={predicted:<8} "
        f"[{status}]"
    )

detected = np.sum(predictions == 1)

print()
print("========================================")
print(" ADAPTED MODEL RESULT")
print("========================================")
print()

print(f"Threshold          : {threshold:.2f}")
print(f"VIKRAM detected    : {detected} / {len(X)}")
print(f"Detection rate     : {detected / len(X) * 100:.2f}%")
print()

print("DONE")