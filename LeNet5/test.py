import torch
import torchvision as tv
import torchvision.transforms as transforms
from model import LeNet5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
testset = tv.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Load model
model = LeNet5().to(device)
model.load_state_dict(torch.load("lenet5_cifar10.pth"))
model.eval()

# Testing
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