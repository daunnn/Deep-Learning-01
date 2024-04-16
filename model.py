import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.swish = Swish() 

    def forward(self, img):
        x = self.swish(self.conv1(img))
        x = torch.max_pool2d(x, kernel_size=2)
        x = self.swish(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = self.swish(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.swish(self.fc1(x))
        x = self.swish(self.fc2(x))
        return x

    def summary(self):
        print("LeNet5 Summary:")
        summary(self, (1, 28, 28))
    

class LeNet5_regularization(nn.Module):
    
    def __init__(self):
        super(LeNet5_regularization, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)
        self.conv3_bn = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120, 84)
        self.fc1_bn = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.2)
        self.swish = Swish() 

    def forward(self, img):
        x = self.swish(self.conv1_bn(self.conv1(img)))
        x = torch.max_pool2d(x, kernel_size=2)
        x = self.swish(self.conv2_bn(self.conv2(x)))
        x = torch.max_pool2d(x, kernel_size=2)
        x = self.swish(self.conv3_bn(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.swish(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.swish(self.fc2(x))
        return x
    
    def summary(self):
        print("LeNet5_regularization Summary:")
        summary(self, (1, 28, 28))
    

class CustomMLP(nn.Module):
    """ Your custom MLP model
    - Note that the number of model parameters should be about the same
    with LeNet-5
    """
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.swish = Swish() 

    def forward(self, img):
        x = torch.flatten(img, 1)
        x = self.swish(self.fc1(x))
        x = self.swish(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def summary(self):
        print("CustomMLP Summary:")
        summary(self, (1, 28, 28))


