"""
fit a model to the training data.
You are allowed to use numpy, scipy.optimize and torch 
only.
"""


from __future__ import annotations
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math

NDArrayFloat = NDArray[np.float_]

device = torch.device("cuda") if torch.cuda.is_available() else torch.cuda.device("cpu")
print("Device is: ", str(device))


class CustomDataset(Dataset):
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
        n_x = len(
            np.unique(self.x)
        )  # why use unique as "parachute"? Shouldn't have two data points with equal value if the data has been cleaned thoroughly
        n_y = len(
            np.unique(self.y)
        )  # why use unique as "parachute"? Shouldn't have two data points with equal value if the data has been cleaned thoroughly
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        conts = ax.pcolormesh(
            self.x.reshape(n_y, n_x)[0, :],
            self.y.reshape(n_y, n_x)[:, 0],
            self.target.reshape(n_y, n_x),
        )
        fig.colorbar(conts)
        plt.tight_layout()
        plt.show()

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        sample = {
            "input_data": torch.tensor(
                np.stack((self.x[index], self.y[index]), axis=0)
            ),
            "target": torch.tensor(self.target[index]),
        }

        return sample

    def __len__(self) -> int:
        return self.target.shape[0]


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.fc1 = nn.Linear(in_features=2, out_features=10, dtype=torch.float64)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=10, out_features=1, dtype=torch.float64)

    def forward(self, x):

        x = x.double()
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)

        return out

    def predict(self, x: np.ndarray, y: np.ndarray) -> list:
        zipped_inputs = list(zip(x, y))
        stacked_inputs = [
            torch.tensor(np.stack(tup, axis=0)).to(device=device)
            for tup in zipped_inputs
        ]
        predictions = [None] * len(stacked_inputs)

        with torch.no_grad():
            for idx, input_sample in enumerate(stacked_inputs):
                input_sample.to(device)
                predictions[idx] = self.forward(input_sample)

        return predictions


def plot_curves(
    y: np.ndarray,
    y_axis_title: str,
    y_1_title: str,
    plot_title: str,
    x: np.ndarray = None,
    y_2: np.ndarray = None,
    y_2_title: str = None,
    x_axis_title: str = None,
) -> None:
    if not x:
        x = np.array([i for i in range(0, y.shape[0])])
        print("x-y-length-check OK? -- ", x.shape[0] == y.shape[0])
    if not x_axis_title:
        x_axis_title = "index"

    plt.title(plot_title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.plot(x, y, label=y_1_title)
    if y_2 is not None and y_2_title is not None:
        plt.plot(x, y_2, label=y_2_title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def model_training(train_dataset: CustomDataset):

    print("\n============")
    print("M O D E L   T R A I N I N G ...")

    print("Creating training dataloader...")
    data_loader_train = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    print("Plotting training data...")
    train_dataset.plot()

    print("Creating model object...")
    model = MyModel()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0002)

    loss_fn = nn.MSELoss()

    num_epochs = 100
    print_interval_epochs = 5

    loss_train_avg = []

    model.to(device=device)

    print("Starting model training process...")
    for epoch in range(0, num_epochs):
        loss_train_cur = 0
        loss_train_cumulated = 0

        model.train()

        for idx_batch, data_batch in enumerate(data_loader_train):

            input_data, target_data = data_batch["input_data"].to(device), data_batch[
                "target"
            ].to(device)

            optimizer.zero_grad()
            predictions = model(input_data)

            loss_train_cur = loss_fn(predictions, target_data)
            loss_train_cur.backward()
            optimizer.step()

            loss_train_cumulated += loss_train_cur.item()

        model.eval()
        loss_train_avg.append(loss_train_cumulated / len(data_loader_train))

        if epoch % print_interval_epochs == 0:
            print(f"-- epoch:{epoch}, train loss: {loss_train_avg[-1]}")
    print("-- Finished training.")

    print("Plotting training loss curve...")
    plot_curves(
        y=np.array(loss_train_avg),
        y_1_title="training loss",
        plot_title="Losses",
        y_axis_title="loss",
        x_axis_title="epoch",
    )

    return model


def calculate_RMSE(target: np.ndarray, predictions: np.ndarray):
    RMSE = math.sqrt(
        sum([(pred - tar) ** 2 for tar, pred in zip(list(target), list(predictions))])
        / target.shape[0]
    )
    return RMSE


if __name__ == "__main__":

    print("Reading training data from csv file...")
    train_dataset = CustomDataset.from_dump(Path("train_dataset.csv"))
    trained_model = model_training(train_dataset=train_dataset)

    train_predictions = trained_model.predict(x=train_dataset.x, y=train_dataset.y)
    np_train_predictions = np.array(
        [p.detach().cpu().numpy() for p in train_predictions]
    )
    RMSE_train = calculate_RMSE(
        target=train_dataset.target, predictions=np_train_predictions
    )
    print("TRAIN data RMSE: ", RMSE_train)

    print("\nReading TEST data from csv files...")
    test_dataset = CustomDataset.from_dump(Path("test_dataset.csv"))
    print("Plotting the test data...")
    test_dataset.plot()

    print("Making TEST data predictions using trained model...")
    test_predictions = trained_model.predict(x=test_dataset.x, y=test_dataset.y)
    np_test_predictions = np.array([p.detach().cpu().numpy() for p in test_predictions])
    RMSE_test = calculate_RMSE(
        target=test_dataset.target, predictions=np_test_predictions
    )
    print("Test data RMSE: ", RMSE_test)
