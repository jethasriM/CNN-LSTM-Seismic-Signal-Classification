import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_and_preprocess
from model.cnn_lstm_model import build_model

from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ------------------------------
# 1. Creating output directories
# ------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ------------------------------
# 2. Loading the data
# ------------------------------

X, y = load_and_preprocess(
    "data/SeisTask_data.h5",
    "data/SeisTask_metadata.csv"
)

metadata = pd.read_csv(
    "data/SeisTask_metadata.csv"
)

groups = metadata["task_id"].values


# ------------------------------
# 3. Train / validation / test split
# ------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(X, y, groups=groups)
)


splitter_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

temp_val_idx, temp_test_idx = next(
    splitter_test.split(
        X[temp_idx],
        y[temp_idx],
        groups=groups[temp_idx]
    )
)

val_idx = temp_idx[temp_val_idx]
test_idx = temp_idx[temp_test_idx]


# ------------------------------
# 4. Creating datasets
# ------------------------------

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]


print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# --------------------------------------
# 5. Normalized using training data only
# --------------------------------------

mean = X_train.mean(
    axis=(0, 1),
    keepdims=True
)

std = X_train.std(
    axis=(0, 1),
    keepdims=True
)

std = np.where(
    std == 0,
    1,
    std
)


X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


print("\nNormalization:")
print("Mean:", mean.flatten())
print("Std:", std.flatten())


# --------------------------------
# 6. Save normalization parameters
# --------------------------------

np.savez(
    "models/normalization.npz",
    mean=mean,
    std=std
)

print("\nNormalization parameters saved.")


# ------------------------------
# 7. Building the model
# ------------------------------

model = build_model(
    X_train.shape[1:]
)

model.summary()


# ------------------------------
# 8. Callbacks
# ------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/earthquake_model.keras",
    monitor="val_loss",
    save_best_only=True
)


# ------------------------------
# 9. Train
# ------------------------------

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=15,
    batch_size=128,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1
)


# ------------------------------
# 10. Final test evaluation
# ------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


print("\n------------------------------")
print("FINAL TEST RESULTS")
print("------------------------------")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# ------------------------------
# 11. Save training history
# ------------------------------

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "models/training_history.csv",
    index=False
)

print("\nTraining history saved.")


# ------------------------------
# 12. Plot training history
# ------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN-LSTM Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "plots/training_accuracy.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN-LSTM Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "plots/training_loss.png",
    dpi=300
)

plt.close()


print("Training plots saved.")