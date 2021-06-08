# import torch.nn.functional as F
from torch import nn

class NeuralNetwork(nn.Module):

    # Initializes the Neural Network by setting up the layers.
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()                       
        self.input_layer = nn.Sequential(nn.Linear(28*28, 512), nn.ReLU()) #first layer has 784 input values, and 512 output values
        self.hidden_layer1 = nn.Linear(512, 128)
        self.hidden_layer2 = nn.Linear(128, 10)#output size is 10, because we expect a number from 0 to 9.
        self.output_layer = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return x