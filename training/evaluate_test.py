import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"

MODEL_PATH = os.path.join(BASE, "vikram_cnn.keras")
FEATURES_PATH = os.path.join(BASE, "features_test.npz")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading test features...")
data = np.load(FEATURES_PATH)

X_test = data["X"]
y_test = data["y"]

X_test = X_test[..., np.newaxis]

print()
print("Test samples:", len(y_test))
print("Feature shape:", X_test.shape)
print()

# Predict
probabilities = model.predict(X_test, verbose=0).flatten()

# Convert probability to class
y_pred = (probabilities >= 0.5).astype(int)

# Class names
# 1 = VIKRAM
# 0 = NEGATIVE

print("====================================")
print("INDIVIDUAL TEST PREDICTIONS")
print("====================================")

# Reconstruct the same test-file order used during feature extraction
test_names = []

mohana_folder = r"C:\Users\lokes\Desktop\dataset\vikram_ml\test\mohana"
negative_folder = r"C:\Users\lokes\Desktop\dataset\vikram_ml\test\negative"

for filename in sorted(os.listdir(mohana_folder)):
    if filename.lower().endswith(".wav"):
        test_names.append(("VIKRAM", filename))

for filename in sorted(os.listdir(negative_folder)):
    if filename.lower().endswith(".wav"):
        test_names.append(("NEGATIVE", filename))

for i, (actual_name, filename) in enumerate(test_names):
    actual = y_test[i]
    predicted = y_pred[i]
    probability = probabilities[i]

    predicted_name = "VIKRAM" if predicted == 1 else "NEGATIVE"

    status = "CORRECT" if actual == predicted else "WRONG"

    print(
        f"{i+1:02d}. {filename:<25} "
        f"Actual={actual_name:<8} "
        f"Predicted={predicted_name:<8} "
        f"Confidence={probability:.4f} "
        f"[{status}]"
    )

print()
print("====================================")
print("CONFUSION MATRIX")
print("====================================")

cm = confusion_matrix(y_test, y_pred)

print()
print("              Predicted")
print("             NEG   VIKRAM")
print(f"Actual NEG   {cm[0,0]:3d}    {cm[0,1]:3d}")
print(f"Actual VIK   {cm[1,0]:3d}    {cm[1,1]:3d}")

print()
print("====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["NEGATIVE", "VIKRAM"],
        zero_division=0
    )
)

accuracy = np.mean(y_test == y_pred)

print("====================================")
print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("====================================")