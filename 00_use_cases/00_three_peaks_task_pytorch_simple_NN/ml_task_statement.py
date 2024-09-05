"""
Given train and test datasets, fit a model to the training data. Choose the most suitable 
model and loss function by yourself. You are allowed to use numpy, scipy.optimize and torch 
only. You can use method Dataset.plot to visualize a dataset. 
Train dataset: https://drive.google.com/file/d/1hyp4EvWKsz2TLQJSnedktqOTHowhMxw7 
Test dataset: https://drive.google.com/file/d/1_NpRRQ4nBe1VxsCJqSJT6rCZ8_dj6L8a 
To solve the problem, fill the following code in:"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd
import torch

NDArrayFloat = NDArray[np.float_]

class Model(Protocol):
    def predict(self, x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:
        ...

class Dataset:
    def __init__(self, target: NDArrayFloat, x: NDArrayFloat, y: NDArrayFloat) -> None:
        self.target = target
        self.x: NDArrayFloat = x
        self.y: NDArrayFloat = y

    @classmethod
    def from_dump(cls, path_to_dataset: Path) -> Dataset:
        df = pd.read_csv(path_to_dataset)
        return cls(
            target=df["target"].to_numpy(), x=df["x"].to_numpy(), y=df["y"].to_numpy()
        )

    def plot(self) -> None:
        n_x = len(np.unique(self.x))
        n_y = len(np.unique(self.y))
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        conts = ax.pcolormesh(
            self.x.reshape(n_y, n_x)[0, :],
            self.y.reshape(n_y, n_x)[:, 0],
            self.target.reshape(n_y, n_x),
        )
        fig.colorbar(conts)
        plt.tight_layout()
        plt.show()


def fit_three_peaks(train_dataset: Dataset) -> Model:

    ##########################
    ### PUT YOUR CODE HERE ###
    ##########################

    return ...


if __name__ == "__main__":
    train_dataset = d = Dataset.from_dump(Path("train_dataset.csv"))
    test_dataset = Dataset.from_dump(Path("test_dataset.csv"))

    m = fit_three_peaks(train_dataset)
    pred = m.predict(test_dataset.x, test_dataset.y)
    print(pred)