import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 30 * 30, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x