import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet5(nn.Module):
  def __init__(self, in_channels=3, num_classes=10):
    super(Lenet5, self).__init__()
    self.in_channels = in_channels
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(self.in_channels, 6*in_channels, kernel_size=5)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 =nn.Conv2d(6*in_channels, 16*in_channels, kernel_size=5)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(16*5*5*self.in_channels, 120*self.in_channels)
    self.fc2 = nn.Linear(120*self.in_channels, 84*self.in_channels)
    self.fc3 = nn.Linear(84*self.in_channels, self.num_classes)

  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  