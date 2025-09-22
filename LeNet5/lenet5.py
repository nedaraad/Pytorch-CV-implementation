import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np


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
  
trainset = tv.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = tv.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Lenet5().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training
# ----------------------------
epochs = 30
for epoch in range(epochs):
  training_loss = 0.0
  correct = 0
  total = 0
  for batch_idx, data in enumerate(trainloader, 0):
    input, label = data[0].to(device), data[1].to(device)
    optimizer.zero_grad()
    pred =model(input)
    loss = criterion(pred, label)
    loss.backward()
    optimizer.step()

    training_loss += loss.item()
    _, predicted = torch.max(pred.data, 1)
    total += data[1].size(0)
    correct += (predicted == label).sum().item()

    if batch_idx % 100 == 99:
      avg_loss = training_loss /100
      avg_acc = 100 * correct / total
      print(f"[Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}] "f"Loss: {avg_loss: .3f} | Accuracy: {avg_acc:.2f}%")
      training_loss = 0.0
      correct = 0
      total = 0
print("Finished Training!")


# Testing
# -----------------------------
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on 10000 test images: {100 * correct / total:.2f}%")