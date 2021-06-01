import os, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose

class NumberMachine(nn.Module):

    def __init__(self):
        super(NumberMachine, self).__init__()
        print('Number machine is running')
        # Initializes training and test data sets.
        self.training_data = datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]), batch_size=64, shuffle=True)
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]), batch_size=64, shuffle=True)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), #first layer has 784 input values, and 512 output values
            nn.ReLU(),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, 10), #output size is 10, because we expect a number from 0 to 9.
            nn.LogSoftmax(1))

    # Checks if Nvidias cuda is available.
    def check_cuda_availability(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device))

    def save_model(self):
        torch.save(model, './my_trained_mnist_model.pt')
        print('Model is saved')

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
