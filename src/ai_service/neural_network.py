from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):

    # Initializes the Neural Network by setting up the layers.
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()                       
        self.linear_layer = nn.Sequential(nn.Linear(28 * 28, 512), nn.ReLU(), #first layer has 784 input values (pixels), and 512 output values
        nn.Linear(512, 254), nn.ReLU(), 
        nn.Linear(254, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10)) # output size is 10, because we expect a number from 0 to 9.
        
    # Forward function is the calculation process, it computes output values 
    # from input values. Activation function(Input * weight + bias)
    # Activation function -> ReLU 
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x