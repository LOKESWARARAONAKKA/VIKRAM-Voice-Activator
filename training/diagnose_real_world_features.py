import numpy as np
import os

FEATURES_REAL = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_real_world.npz"
FEATURES_TRAIN = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_train.npz"

real = np.load(FEATURES_REAL)
train = np.load(FEATURES_TRAIN)

X_real = real["X"]
y_real = real["y"]
names = real["names"]

X_train = train["X"]
y_train = train["y"]

print("========================================")
print(" REAL-WORLD FEATURE DIAGNOSTIC")
print("========================================")
print()

# Separate real-world samples
real_vikram = X_real[y_real == 1]
real_negative = X_real[y_real == 0]

# Separate training samples
train_vikram = X_train[y_train == 1]
train_negative = X_train[y_train == 0]

print("REAL-WORLD")
print("----------")
print("VIKRAM samples :", len(real_vikram))
print("Negative samples:", len(real_negative))
print()

print("TRAINING")
print("--------")
print("VIKRAM samples :", len(train_vikram))
print("Negative samples:", len(train_negative))
print()

print("========================================")
print("OVERALL FEATURE STATISTICS")
print("========================================")

print()
print("Real VIKRAM")
print("Mean :", np.mean(real_vikram))
print("Std  :", np.std(real_vikram))

print()
print("Real Negative")
print("Mean :", np.mean(real_negative))
print("Std  :", np.std(real_negative))

print()
print("Train VIKRAM")
print("Mean :", np.mean(train_vikram))
print("Std  :", np.std(train_vikram))

print()
print("Train Negative")
print("Mean :", np.mean(train_negative))
print("Std  :", np.std(train_negative))

print()
print("========================================")
print("PER-FILE REAL-WORLD STATISTICS")
print("========================================")

for i, name in enumerate(names):

    mean_value = np.mean(X_real[i])
    std_value = np.std(X_real[i])
    min_value = np.min(X_real[i])
    max_value = np.max(X_real[i])

    label = "VIKRAM" if y_real[i] == 1 else "NEGATIVE"

    print(
        f"{name:<25} "
        f"{label:<9} "
        f"mean={mean_value:8.3f} "
        f"std={std_value:8.3f} "
        f"min={min_value:8.3f} "
        f"max={max_value:8.3f}"
    )

print()
print("========================================")
print("DIAGNOSTIC COMPLETE")
print("========================================")