from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    LSTM,
    Dense,
    Dropout
)


def build_model(input_shape):

    model = Sequential([
        Input(shape=input_shape),

        # CNN feature extraction
        Conv1D(
            filters=32,
            kernel_size=7,
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),

        Conv1D(
            filters=64,
            kernel_size=5,
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),

        MaxPooling1D(pool_size=2),

        Dropout(0.2),

        # Temporal dependency learning
        LSTM(
            64,
            return_sequences=False
        ),

        Dropout(0.3),

        # Classification head
        Dense(
            32,
            activation="relu"
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