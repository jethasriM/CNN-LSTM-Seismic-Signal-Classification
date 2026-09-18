from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout
)


def build_model(input_shape):

    model = Sequential([
        Input(shape=input_shape),

        Conv1D(
            filters=32,
            kernel_size=7,
            activation="relu"
        ),

        MaxPooling1D(
            pool_size=4
        ),

        Dropout(0.2),

        LSTM(
            32,
            return_sequences=False
        ),

        Dropout(0.2),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model