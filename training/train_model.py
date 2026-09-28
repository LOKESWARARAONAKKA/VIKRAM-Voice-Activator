import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# VIKRAM - CNN TRAINING
# ============================================================

TRAIN_FILE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_train.npz"
VALIDATION_FILE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_validation.npz"
TEST_FILE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\features_test.npz"

MODEL_FILE = r"C:\Users\lokes\Desktop\VIKRAM-Voice-Activator\training\vikram_cnn.keras"

# ============================================================
# LOAD DATA
# ============================================================

print()
print("============================================")
print(" VIKRAM CNN TRAINING")
print("============================================")
print()

train_data = np.load(TRAIN_FILE)
validation_data = np.load(VALIDATION_FILE)
test_data = np.load(TEST_FILE)

X_train = train_data["X"]
y_train = train_data["y"]

X_validation = validation_data["X"]
y_validation = validation_data["y"]

X_test = test_data["X"]
y_test = test_data["y"]

print("Train data:", X_train.shape)
print("Validation data:", X_validation.shape)
print("Test data:", X_test.shape)

print()
print("Train VIKRAM:", np.sum(y_train == 1))
print("Train Negative:", np.sum(y_train == 0))

print("Validation VIKRAM:", np.sum(y_validation == 1))
print("Validation Negative:", np.sum(y_validation == 0))

print("Test VIKRAM:", np.sum(y_test == 1))
print("Test Negative:", np.sum(y_test == 0))

# ============================================================
# PREPARE DATA FOR CNN
# ============================================================

# CNN expects:
# samples, height, width, channels

X_train = X_train[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print()
print("CNN input shape:", X_train.shape)

# ============================================================
# BUILD SMALL CNN
# ============================================================

model = models.Sequential([
    
    layers.Input(
        shape=(40, 151, 1)
    ),

    layers.Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        32,
        activation="relu"
    ),

    layers.Dropout(
        0.3
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])

# ============================================================
# DISPLAY MODEL
# ============================================================

print()
print("============================================")
print(" MODEL SUMMARY")
print("============================================")

model.summary()

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# ============================================================
# TRAIN
# ============================================================

print()
print("============================================")
print(" STARTING TRAINING")
print("============================================")
print()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_validation,
        y_validation
    ),
    epochs=50,
    batch_size=8,
    callbacks=[
        early_stopping
    ],
    verbose=1
)

# ============================================================
# VALIDATION RESULT
# ============================================================

print()
print("============================================")
print(" VALIDATION RESULT")
print("============================================")

validation_result = model.evaluate(
    X_validation,
    y_validation,
    verbose=0
)

for name, value in zip(
    model.metrics_names,
    validation_result
):
    print(
        f"{name}: {value:.4f}"
    )

# ============================================================
# TEST RESULT
# ============================================================

print()
print("============================================")
print(" TEST RESULT")
print("============================================")

test_result = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

for name, value in zip(
    model.metrics_names,
    test_result
):
    print(
        f"{name}: {value:.4f}"
    )

# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_FILE
)

print()
print("============================================")
print(" TRAINING COMPLETE")
print("============================================")

print()
print("Model saved to:")
print(MODEL_FILE)

print()