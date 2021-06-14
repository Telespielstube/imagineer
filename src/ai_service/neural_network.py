# import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):

    # Initializes the Neural Network by setting up the layers.
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()                       
        self.input_layer = nn.Sequential(nn.Linear(28*28, 512)) #first layer has 784 input values, and 512 output values
        self.hidden_layer1 = nn.Linear(512, 254)
        self.hidden_layer2 = nn.Linear(254, 128)#output size is 10, because we expect a number from 0 to 9.
        self.hidden_layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    # Forward function is the calculation process, it computes output values 
    # from input values. Activation function(Input * weight + bias)
    # Activation function -> ReLU 
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.hidden_layer3(x))
        x = self.output_layer(x)
        
        return F.log_softmax(x, 1)