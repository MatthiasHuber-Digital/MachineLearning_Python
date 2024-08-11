import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("CUDA Device is: ", device)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform,
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True,
)  # potentially reduce batch size
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False,
)

model_vgg16 = models.vgg16(pretrained=True)
model_vgg16.to(device)

print(model_vgg16)

# CHANGE THE NUMBER OF OUTPUT CLASSES TO 10
model_vgg16.classifier[-1].out_features = 10

# Freeze the parameters using .requires_grad
for param in model_vgg16.features.parameters():
    param.requires_grad = False

optimizer = torch.optim.SGD(
    params=model_vgg16.classifier.parameters(), lr=0.001, momentum=0.9,
)

loss_callable = nn.CrossEntropyLoss()


def model_computation_single_epoch(
    model, dataloader: torch.utils.data.DataLoader, train: bool,
):
    
    if train:
        model.train()  # Gradient updates active - backprop ON - tune weights using gradient descent
    else:
        model.eval()  # SHUTS OFF BACKPROPAGATION OF LOSS AND THUS GRADIENT COMPUTATION / GRADIENT DESCENT

    loss_curr = 0
    running_loss = 0.0
    correct_predictions = 0

    for idx, data in enumerate(dataloader):

        if idx % 20 == 0:
            print("------batch: ", str(idx))
        X, y_ground_truth = data[0].to(device), data[1].to(device)
        if train:
            optimizer.zero_grad()  # gradients for every mini-batch should be zero in the beginning. This is NOT necessary for RNNs where you need to accumulate the gradient.

        y_pred_probs = model(X)

        loss_curr = loss_callable(input=y_pred_probs, target=y_ground_truth)
        running_loss += loss_curr.item()

        _, y_pred_classes = torch.max(y_pred_probs.data, 1) # limit the prediction probability to 1
        correct_predictions += (y_pred_classes == y_ground_truth).sum().item()

        if train:
            loss_curr.backward()
            optimizer.step()

    loss_epoch = running_loss / len(dataloader.dataset)
    accuracy_epoch = 100 * correct_predictions / len(dataloader.dataset)

    return loss_epoch, accuracy_epoch


train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []
start_time = time.time()

num_epochs = 1

print("Start training model_vgg16...")
for epoch in range(0, num_epochs):
    train_loss_curr, train_accuracy_curr = model_computation_single_epoch(model=model_vgg16, dataloader=train_dataloader, train=True)
    print("--epoch: ", epoch, ", train loss: ", round(train_loss_curr, 6))

    train_accuracy.append(train_accuracy_curr)
    train_loss.append(train_loss_curr)

    if epoch % 3 == 0:
        val_loss_curr, val_accuracy_curr = model_computation_single_epoch(model=model_vgg16, dataloader=train_dataloader, train=False)
        print("--epoch: ", epoch, ", VAL loss: ", round(val_loss_curr, 6))

        val_accuracy.append(val_accuracy_curr)
        val_loss.append(val_loss_curr)


end_time = time.time()

time = str((end_time-start_time)/60)
print('Training finished. It took [mins]:' + time)
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validation accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

save_path = 'vgg16_transfer_learning_CIFAR10_1epoch.pth'

torch.save(model_vgg16, save_path)

try:
    test_model = torch.load(save_path)
except Exception:
    raise ImportError("Failed to save and reimport the trained model.")
else:
    print("Successfully saved the model to the disk.")