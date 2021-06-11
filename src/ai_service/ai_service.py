import torch 
import numpy as np
from time import time
from ai_service.neural_network import NeuralNetwork
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose
class AiService():

    def __init__(self, save_path):
        self.batch_size = 200
        self.epochs = 5
        self.learning_rate = 0.01
        self.training_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
        self.validation_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
        self.path = save_path
        self.model = NeuralNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Function to train the mnist dataset.
    def training_phase(self):
        criterion = nn.CrossEntropyLoss() #combines LogSoftmax and NLLLoss in one single class.
        optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        start_time = time()
        for epoch in range(self.epochs):
            running_loss = 0
            # trainig phase
            for images, labels in self.training_data:
                optimizer.zero_grad() 
                image, label = image.to(self.device), label.to(self.device)
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward() #This is where the model learns by backpropagating
                optimizer.step() #optimizing weights
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {:.10f}".format(epoch, running_loss / len(self.training_data)))
        print("\nTraining Time (in minutes): {:.0f} =".format((time() - start_time) / 60))

    # Function validates the trained model against the received image.
    # @cv_image    cv_image image object to be validated.
    # @return      a predicted number. 
    def validation_phase(self, cv_image):
        self.model.eval()
        tensor_image = self.image_to_tensor(cv_image)
        #image = tensor_image[0].view(1, 28, 28)
        with torch.no_grad():
            output = self.model(tensor_image) # model returns the vector of raw predictions that a classification model generates.         
        ps = torch.exp(output)
        probability = list(ps.numpy()) # a list of possible numbers
        return probability.index(max(probability))
    
    # Uses the standard MNIST validation data set to test the trained model.
    def mnist_validation(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for image, label in self.validation_data:
                image, label = image.to(self.device), label.to(self.device)
                output = self.model(image)
                test_loss += criterion(output, label).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(self.validation_data.dataset)
        print('\n Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.validation_data.dataset), 100. * correct / len(self.validation_data.dataset)))

    # Saves the entire trained model to a specific path.
    def save_model(self):
        torch.save(self.model, self.path)
        print('Model is saved')
    
    # Loads entire saved model.
    def load_model(self):
       self.model = torch.load(self.path)

    # Converts the image which is respresnted as numpy array to a PyTorch readable tensor.
    # @cv_image    Image object in opencv format.
    #
    # @return      cv_image converted to PyTorch tensor.
    def image_to_tensor(self, cv_image):
        return transforms.ToTensor()(cv_image)