import h5py
import pandas as pd


with h5py.File(
    "data/SeisTask_data.h5",
    "r"
) as f:

    data = f["data"][:]


metadata = pd.read_csv(
    "data/SeisTask_metadata.csv"
)


# Find one signal sample
signal_index = metadata[
    metadata["signal"] == 1
].index[0]


# Find one noise sample
noise_index = metadata[
    metadata["signal"] == 0
].index[0]


# Convert:
# (2, 500)
# →
# (500, 2)

signal = data[
    signal_index
].T

noise = data[
    noise_index
].T


# Save CSVs

pd.DataFrame(
    signal,
    columns=[
        "channel_1",
        "channel_2"
    ]
).to_csv(
    "sample_signal.csv",
    index=False
)


pd.DataFrame(
    noise,
    columns=[
        "channel_1",
        "channel_2"
    ]
).to_csv(
    "sample_noise.csv",
    index=False
)


print(
    "Created sample_signal.csv"
)

print(
    "Created sample_noise.csv"
)

print(
    "Signal index:",
    signal_index
)

print(
    "Noise index:",
    noise_index
)