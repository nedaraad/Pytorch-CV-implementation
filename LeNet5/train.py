import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Lenet5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
trainset = tv.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Model, loss, optimizer
model = Lenet5().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training loop
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
torch.save(model.state_dict(), "lenet5_cifar10.pth")



