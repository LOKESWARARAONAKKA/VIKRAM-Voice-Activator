import numpy as np
from tensorflow.keras.models import load_model

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"

MODEL_PATH = BASE + r"\vikram_cnn.keras"
FEATURE_PATH = BASE + r"\features_real_world.npz"

print("========================================")
print(" REAL-WORLD MODEL TEST")
print("========================================")

print("\nLoading model...")
model = load_model(MODEL_PATH)

print("Loading real-world features...")
data = np.load(FEATURE_PATH)

X = data["X"]
y = data["y"]

X = X[..., np.newaxis]

print(f"Samples: {len(X)}")
print(f"Feature shape: {X.shape}")

probabilities = model.predict(X, verbose=0).flatten()

print("\n========================================")
print(" INDIVIDUAL RESULTS")
print("========================================")

for i, probability in enumerate(probabilities):
    actual = "VIKRAM" if y[i] == 1 else "NEGATIVE"
    predicted = "VIKRAM" if probability >= 0.90 else "NEGATIVE"

    correct = "YES" if predicted == actual else "NO"

    print(
        f"{i+1:02d}. "
        f"Actual={actual:<8} "
        f"Confidence={probability:.10f} "
        f"Predicted={predicted:<8} "
        f"Correct={correct}"
    )

predictions = (probabilities >= 0.90).astype(int)

accuracy = np.mean(predictions == y)

false_positives = np.sum((predictions == 1) & (y == 0))
false_negatives = np.sum((predictions == 0) & (y == 1))

print("\n========================================")
print(" REAL-WORLD SUMMARY")
print("========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

print("\nDONE")