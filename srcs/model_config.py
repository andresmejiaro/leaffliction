import torch
from torch import nn

class ImageMLP(nn.Module):
    def __init__(self, in_shape, num_classes, p_drop=0.2):
        super().__init__()
        c,h,w = in_shape
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32,64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),
            nn.Linear(8*8*256, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(200, num_classes)
        )

    def forward(self, x):
        return self.net(x)

