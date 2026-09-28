import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"

MODEL_PATH = BASE + r"\vikram_cnn.keras"
TEST_PATH = BASE + r"\features_validation.npz"

print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading test features...")
data = np.load(TEST_PATH)

X_test = data["X"]
y_test = data["y"]

X_test = X_test[..., np.newaxis]

probabilities = model.predict(X_test, verbose=0).flatten()

print("\n====================================")
print("THRESHOLD TEST")
print("====================================")

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    predictions = (probabilities >= threshold).astype(int)

    accuracy = np.mean(predictions == y_test)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    print(
        f"Threshold={threshold:.2f} | "
        f"Accuracy={accuracy*100:.2f}% | "
        f"False Positives={fp} | "
        f"False Negatives={fn}"
    )

print("\n====================================")
print("INDIVIDUAL TEST PROBABILITIES")
print("====================================")

for i, probability in enumerate(probabilities):
    actual = "VIKRAM" if y_test[i] == 1 else "NEGATIVE"
    print(
        f"{i+1:02d}. Actual={actual:<8} "
        f"VIKRAM_Confidence={probability:.4f}"
    )

print("\nDONE")