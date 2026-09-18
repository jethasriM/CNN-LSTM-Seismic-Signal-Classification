from preprocessing import load_and_preprocess
import pandas as pd

X, y = load_and_preprocess(
    "data/SeisTask_data.h5",
    "data/SeisTask_metadata.csv"
)

print("\nFinal checks:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("First waveform:", X[0].shape)
print("First label:", y[0])
print("Unique labels:", set(y))

metadata = pd.read_csv("data/SeisTask_metadata.csv")

print("\nClass distribution:")
print(metadata["signal"].value_counts())

print("\nClass percentages:")
print(metadata["signal"].value_counts(normalize=True) * 100)