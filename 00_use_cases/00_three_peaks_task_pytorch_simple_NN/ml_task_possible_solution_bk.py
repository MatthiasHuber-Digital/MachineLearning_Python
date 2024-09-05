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

NDArrayFloat = NDArray[np.float_]


class Model(Protocol):
    def predict(self, x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat: ...


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
    
    def predict():
        pass


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
    #fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.title(plot_title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.plot(x, y, label=y_1_title)
    if y_2 is not None and y_2_title is not None:
        plt.plot(x, y_2, label=y_2_title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def model_training():

    print("\n============")
    print("M O D E L   T R A I N I N G ...")

    print("Reading data from csv files...")
    train_dataset = CustomDataset.from_dump(Path("train_dataset.csv"))
    test_dataset = CustomDataset.from_dump(Path("test_dataset.csv"))
    print("Creating dataloaders...")
    data_loader_train = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    data_loader_test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print("Plotting data...")
    # train_dataset.plot()
    # test_dataset.plot()

    print("Creating model object...")
    model = MyModel()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.cuda.device("cpu")
    )
    print("Device is: ", str(device))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    loss_fn = nn.MSELoss()

    num_epochs = 10
    print_interval_epochs = 5

    loss_train_avg = []
    loss_test_avg = []

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

            # print("Batch shape training: ", input_data.shape)

            optimizer.zero_grad()
            predictions = model(input_data)

            loss_train_cur = loss_fn(predictions, target_data)
            loss_train_cur.backward()
            optimizer.step()

            loss_train_cumulated += loss_train_cur.item()

        model.eval()
        loss_train_avg.append(loss_train_cumulated / len(data_loader_test))

        loss_test_cumulated = 0
        with torch.no_grad():
            for idx_batch, data_batch in enumerate(data_loader_test):
                input_data, target_data = data_batch["input_data"].to(
                    device
                ), data_batch["target"].to(device)
                predictions = model(input_data)
                loss_test_cumulated += loss_fn(predictions, target_data).item()

        loss_test_avg.append(loss_test_cumulated / len(data_loader_test))

        if epoch % print_interval_epochs == 0:  # print every 2000 mini-batches
            print(
                f"-- epoch:{epoch}, train loss: {loss_train_avg[-1]}, test loss: {loss_test_avg[-1]}"
            )
    print("-- Finished training.")
    print("\nTraining loss averages: ", loss_train_avg)
    print("Test loss averages: ", loss_train_avg)

    print("Plotting loss curves...")
    plot_curves(y=np.array(loss_train_avg), y_1_title="training loss", y_2=np.array(loss_test_avg), y_2_title="test loss", plot_title="Losses", y_axis_title="loss", x_axis_title="epoch")


if __name__ == "__main__":

    model_training()


"""
    m = fit_three_peaks(train_dataset)
    pred = m.predict(test_dataset.x, test_dataset.y)
    print(pred)
"""
