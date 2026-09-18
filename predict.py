import numpy as np
import h5py
from tensorflow.keras.models import load_model


MODEL_PATH = "models/earthquake_model.keras"
NORMALIZATION_PATH = "models/normalization.npz"


def load_artifacts():

    model = load_model(MODEL_PATH)

    normalization = np.load(
        NORMALIZATION_PATH
    )

    # Converting (1, 1, 2) → (2,)
    mean = normalization["mean"].reshape(2)
    std = normalization["std"].reshape(2)

    return model, mean, std


def preprocess_waveform(waveform, mean, std):
    """
    Preparing a single two-channel waveform.

    Input:
        (500, 2)

    Output:
        (1, 125, 2)
    """

    waveform = np.asarray(
        waveform,
        dtype=np.float32
    )

    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]

    if waveform.shape != (500, 2):
        raise ValueError(
            f"Expected waveform shape (500, 2), "
            f"but received {waveform.shape}"
        )

    waveform = waveform[::4, :]

    # Now:
    # waveform = (125, 2)
    # mean     = (2,)
    # std      = (2,)
    #
    # Result remains:
    # (125, 2)

    waveform = (
        waveform - mean
    ) / std


    waveform = np.expand_dims(
        waveform,
        axis=0
    )

    return waveform


def predict_waveform(waveform):

    model, mean, std = load_artifacts()

    processed = preprocess_waveform(
        waveform,
        mean,
        std
    )

    print(
        "Model input shape:",
        processed.shape
    )

    probability = model.predict(
        processed,
        verbose=0
    )[0][0]

    if probability >= 0.5:
        prediction = "Signal"
        confidence = probability
    else:
        prediction = "Noise"
        confidence = 1 - probability

    return prediction, float(confidence)


if __name__ == "__main__":

    with h5py.File(
        "data/SeisTask_data.h5",
        "r"
    ) as f:

        waveform = f["data"][0]

    # Dataset format:
    # (2, 500)
    #
    # Converting it into:
    # (500, 2)

    waveform = waveform.T

    print(
        "Original waveform shape:",
        waveform.shape
    )

    prediction, confidence = predict_waveform(
        waveform
    )

    print("\n------------------------------")
    print("SEISMIC SIGNAL PREDICTION")
    print("------------------------------")

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : {confidence * 100:.2f}%"
    )