import os, torch
import torch.nn.functional as F
from time import time
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose

class NumberMachine(nn.Module):

    def __init__(self, batch_size, epochs, learning_rate, log_interval):
        super(NumberMachine, self).__init__()
        print('Number machine is running')
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.image = None

        # Initializes training and validation data sets.
        self.training_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), self.batch_size, shuffle=True)
        self.validation_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), self.batch_size, shuffle=True)
        
        # Setup all layers                       
        self.input_layer = nn.Sequential(nn.Linear(28*28, 512), nn.ReLU()) #first layer has 784 input values, and 512 output values
        self.hidden_layer1 = nn.Linear(512, 128), nn.ReLU()
        self.hidden_layer2 = nn.Linear(128, 10), nn.ReLU()#output size is 10, because we expect a number from 0 to 9.
        self.output_layer = nn.LogSoftmax(1)
  
    # Sets the image property attribute.
    # @image        image sent from the controller node. 
    def set_image(self, image):
        self.image = image

    # Core training of the MNIST dataset.
    def train_model(self, model):
        print("Training is running")
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(self.image, self.learning_rate)
        time0 = time()
        for e in range(self.epochs):
            running_loss = 0
            for images, labels in self.training_data:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
            
                # Training pass
                optimizer.zero_grad()
                
                output = model(images)
                loss = criterion(output, labels)
                
                #This is where the model learns by backpropagating
                loss.backward()
                
                #And optimizes its weights here
                optimizer.step()
                
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(self.training_data)))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)

    def save_model(self, model):
        torch.save(model, './my_trained_mnist_model.pt')
        print('Model is saved')

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
