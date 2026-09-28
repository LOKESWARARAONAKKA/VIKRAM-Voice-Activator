import numpy as np

REAL_FILE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_real_world.npz"
TRAIN_FILE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_train.npz"

real = np.load(REAL_FILE)
train = np.load(TRAIN_FILE)

X_real = real["X"]
y_real = real["y"]
names = real["names"]

X_train = train["X"]
y_train = train["y"]

train_vikram = X_train[y_train == 1]
train_negative = X_train[y_train == 0]

print("========================================")
print(" CLASS SIMILARITY DIAGNOSTIC")
print("========================================")
print()

for i, name in enumerate(names):

    sample = X_real[i]

    # Mean absolute difference from every training sample
    diff_vikram = np.mean(
        np.abs(train_vikram - sample),
        axis=(1, 2)
    )

    diff_negative = np.mean(
        np.abs(train_negative - sample),
        axis=(1, 2)
    )

    nearest_vikram = np.min(diff_vikram)
    nearest_negative = np.min(diff_negative)

    avg_vikram = np.mean(diff_vikram)
    avg_negative = np.mean(diff_negative)

    label = "VIKRAM" if y_real[i] == 1 else "NEGATIVE"

    if nearest_vikram < nearest_negative:
        closer = "VIKRAM"
    else:
        closer = "NEGATIVE"

    print(f"{name:<25} {label:<9}")
    print(f"  Nearest VIKRAM  : {nearest_vikram:.4f}")
    print(f"  Nearest Negative: {nearest_negative:.4f}")
    print(f"  Average VIKRAM  : {avg_vikram:.4f}")
    print(f"  Average Negative: {avg_negative:.4f}")
    print(f"  Closer to       : {closer}")
    print()

print("========================================")
print("DIAGNOSTIC COMPLETE")
print("========================================")