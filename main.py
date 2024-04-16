import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LeNet5, CustomMLP, LeNet5_regularization
import dataset
import dataset_regularization
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trn_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    trn_loss = total_loss / len(trn_loader)
    acc = 100. * correct / total
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tst_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    tst_loss = total_loss / len(tst_loader)
    acc = 100. * correct / total
    return tst_loss, acc

def count_parameters(model):
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_losses_and_accuracies(model_name, train_losses, test_losses, train_accuracies, test_accuracies):
    """Plot and save train and test losses and accuracies"""
    epochs = range(1, 16)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.xticks(epochs)

    plt.savefig(f'./plot/{model_name}_loss_plot.png')
    plt.show()

    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracies')
    plt.legend()
    plt.xticks(epochs)

    plt.savefig(f'./plot/{model_name}_accuracy_plot.png')
    plt.show()
    
def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = dataset.MNIST(data_dir='../data/train/')
    test_dataset = dataset.MNIST(data_dir='../data/test/')
    
    ############## Data augmentation 할 경우 #######
#     train_dataset = dataset_regularization.MNIST(data_dir='../data/train/', is_train=True)
#     test_dataset = dataset_regularization.MNIST(data_dir='../data/test/', is_train=False)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Instantiate model
    
    model = LeNet5().to(device)
#     model = CustomMLP().to(device)
#     model = LeNet5_regularization().to(device)

    model.summary()
    print(f"Total Trainable Parameters: {count_parameters(model)}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Lists to store losses and accuracies for plotting
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []


    # Training and Testing
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    model_name = 'LeNet5'  # Change this to model name

    for epoch in range(1, 16):
        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer)
        test_loss, test_acc = test(model, test_loader, device, criterion)

        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    plot_losses_and_accuracies(model_name, train_losses, test_losses, train_accuracies, test_accuracies)
    
    

if __name__ == '__main__':
    main()
