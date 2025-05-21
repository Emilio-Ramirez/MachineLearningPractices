import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_printoptions(linewidth=150, precision=2, threshold=10000)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Training data
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Test data
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Example of a tensor
print(train_dataset.data[0])

#  Data loaders to handle batching
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,  # Process 64 images at a time
    shuffle=True,  # Shuffle the data for better training
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,  # No need to shuffle test data
)


class MNISTNetwork(nn.Module):
    def __init__(self):
        super(MNISTNetwork, self).__init__()

        # Input layer: 784 neurons (28x28 flattened images)
        # Hidden layer: 128 neurons
        self.fc1 = nn.Linear(28 * 28, 128)

        self.fc2 = nn.Linear(128, 64)
        # Hidden layer: 128 neurons
        # Output layer: 10 neurons (one for each digit)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the input image from 28x28 to 784
        x = x.view(-1, 28 * 28)

        # First layer with ReLU activation
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        # Output layer (no activation here - will be applied in the loss function)
        x = self.fc3(x)

        return x


model = MNISTNetwork()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Defie the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    # Iterate over the batches
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels from the data loader
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward class
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}")
            running_loss = 0.0

torch.save(model.state_dict(), "mnist_model.pth")

print("training finish")

# Set the model to evaluation mode
model.eval()

# Variables to track correct predictions and total samples
correct = 0
total = 0

# Disable gradient calculation for evaluation (saves memory and computations)
with torch.no_grad():
    # Iterate through test data
    for data in test_loader:
        # Get inputs and labels
        images, labels = data

        # Get predictions
        outputs = model(images)

        # Get the predicted class (digit with highest score)
        _, predicted = torch.max(outputs.data, 1)

        # Update counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy
accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy:.2f}%")

# To examine performance on each digit class
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Collect results for each class
        c = predicted == labels
        for i in range(len(c)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print per-class accuracy
for i in range(10):
    if class_total[i] > 0:
        print(f"Accuracy of {i}: {100 * class_correct[i] / class_total[i]:.2f}%")
