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

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_dataloader = torch.utils.data.Dataloader(train_dataset, batch_size=32, shuffle=True) # potentially reduce batch size
test_dataloader = torch.utils.data.Dataloader(test_dataset, batch_size=32, shuffle=False)



















print("end")