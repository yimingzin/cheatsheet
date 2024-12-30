import torch
import torchvision
import argparse

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchinfo import summary
from torchvision import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ]),
    download=True
)

mnist_test_data = datasets.MNIST(
    root="data",
    train=False,
    transform=T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ]),
    download=True
)

mnist_train_dataloader = DataLoader(
    mnist_train_data,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

mnist_test_dataloader = DataLoader(
    mnist_test_data,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

class_names = mnist_train_data.classes

class MNISTModel(nn.Module):
    def __init__(self, input_channel: int = 1, hidden_units: int = 10, output_channel: int = len(class_names)):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channel, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_channel)
        )
    
    def forward(self, x):
        y = self.classifier(self.block_2(self.block_1(x)))
        return y

model_mnist = MNISTModel().to(device)

summary(
    model = model_mnist,
    input_size=(32, 1, 28, 28),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)

