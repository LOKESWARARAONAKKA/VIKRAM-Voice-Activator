import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

BASE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training"

MODEL_PATH = os.path.join(BASE, "vikram_cnn.keras")
CONTROLLED_FEATURES = os.path.join(BASE, "features_controlled.npz")
TRAIN_FEATURES = os.path.join(BASE, "features_train.npz")

OUTPUT_MODEL = os.path.join(
    BASE,
    "vikram_cnn_adapted.keras"
)

print("========================================")
print(" VIKRAM MODEL ADAPTATION")
print("========================================")

print("\nLoading original model...")
model = load_model(MODEL_PATH)

print("Loading original training features...")
train_data = np.load(TRAIN_FEATURES)

X_train = train_data["X"]
y_train = train_data["y"]

print("Loading controlled VIKRAM features...")
controlled_data = np.load(CONTROLLED_FEATURES)

X_controlled = controlled_data["X"]
y_controlled = controlled_data["y"]

print()
print("Original training samples :", len(X_train))
print("Controlled samples        :", len(X_controlled))

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_controlled = X_controlled[..., np.newaxis]

# Combine original training data with controlled recordings
X_combined = np.concatenate(
    [X_train, X_controlled],
    axis=0
)

y_combined = np.concatenate(
    [y_train, y_controlled],
    axis=0
)

print("Combined samples          :", len(X_combined))
print("Combined VIKRAM           :", np.sum(y_combined == 1))
print("Combined NEGATIVE         :", np.sum(y_combined == 0))

print()
print("Fine-tuning model...")

# Use a smaller learning rate for adaptation
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00005
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

early_stopping = EarlyStopping(
    monitor="loss",
    patience=8,
    restore_best_weights=True
)

model.fit(
    X_combined,
    y_combined,
    epochs=30,
    batch_size=8,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)

print()
print("Saving adapted model...")

model.save(OUTPUT_MODEL)

print()
print("========================================")
print(" ADAPTATION COMPLETE")
print("========================================")
print()
print("Saved model:")
print(OUTPUT_MODEL)