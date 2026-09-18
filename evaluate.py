import numpy as np
import pandas as pd
import h5py
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

from tensorflow.keras.models import load_model


# ------------------------------
# 1. Loading the metadata
# ------------------------------

metadata = pd.read_csv(
    "data/SeisTask_metadata.csv"
)

y = metadata["signal"].values.astype(
    np.float32
)

groups = metadata["task_id"].values


# ------------------------------
# 2. Loading waveform data
# ------------------------------

with h5py.File(
    "data/SeisTask_data.h5",
    "r"
) as f:

    X = f["data"][:].astype(
        np.float32
    )


# ------------------------------
# 3. Preparing waveform
# ------------------------------

# (samples, channels, time)
# →
# (samples, time, channels)

X = np.transpose(
    X,
    (0, 2, 1)
)

X = X[:, ::4, :]


# -----------------------------------
# 4. Recreating the exact test split
# -----------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
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

test_idx = temp_idx[temp_test_idx]


X_train = X[train_idx]

X_test = X[test_idx]

y_test = y[test_idx]


# ----------------------------------
# 5. Loading training normalization
# ----------------------------------

normalization = np.load(
    "models/normalization.npz"
)

mean = normalization["mean"]
std = normalization["std"]


# ------------------------------
# 6. Normalized the test data
# ------------------------------

X_test = (
    X_test - mean
) / std


# ------------------------------
# 7. Loading the model
# ------------------------------

model = load_model(
    "models/earthquake_model_v2.keras"
)


# ------------------------------
# 8. Generating predictions
# ------------------------------

probabilities = model.predict(
    X_test,
    verbose=1
).flatten()


predictions = (
    probabilities >= 0.5
).astype(int)


# ------------------------------
# 9. Classification report
# ------------------------------

print("\n------------------------------")
print("CLASSIFICATION REPORT")
print("------------------------------")

report = classification_report(
    y_test,
    predictions,
    target_names=[
        "Noise",
        "Signal"
    ]
)

print(report)


# ------------------------------
# 10. Confusion matrix
# ------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n------------------------------")
print("CONFUSION MATRIX")
print("------------------------------")

print(cm)


# ------------------------------
# 11. ROC-AUC
# ------------------------------

auc = roc_auc_score(
    y_test,
    probabilities
)

print("\n------------------------------")
print("ROC-AUC")
print("------------------------------")

print(f"{auc:.4f}")


# --------------------------------------
# 12. Saving the confusion matrix plot
# --------------------------------------

os.makedirs(
    "plots",
    exist_ok=True
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "CNN-LSTM Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    ["Noise", "Signal"]
)

plt.yticks(
    [0, 1],
    ["Noise", "Signal"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "plots/confusion_matrix.png",
    dpi=300
)

plt.close()


# ------------------------------
# 13. Save ROC curve
# ------------------------------

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title(
    "CNN-LSTM ROC Curve"
)

plt.tight_layout()

plt.savefig(
    "plots/roc_curve.png",
    dpi=300
)

plt.close()


print("\nEvaluation plots saved.")
