import numpy as np
import pandas as pd
import h5py


def load_and_preprocess(
    data_path="data/SeisTask_data.h5",
    metadata_path="data/SeisTask_metadata.csv"
):
    metadata = pd.read_csv(metadata_path)

    with h5py.File(data_path, "r") as f:
        X = f["data"][:]

    X = X.astype(np.float32)

    # (samples, channels, time)
    # → (samples, time, channels)
    X = np.transpose(X, (0, 2, 1))
    X = X[:, ::4, :]

    y = metadata["signal"].values.astype(np.float32)

    print("Waveform shape:", X.shape)
    print("Labels shape:", y.shape)

    return X, y