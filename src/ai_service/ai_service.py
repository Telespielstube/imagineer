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
            for images, labels in self.training_data:
                images = images.view(images.shape[0], -1) # Flatten MNIST images into a 784 long vector
                optimizer.zero_grad() # Training pass 
                if torch.cuda.is_available():
                    output = self.model(images.cuda)
                    loss = criterion(output, labels.cuda)  
                else:          
                    output = self.model(images)
                    loss = criterion(output, labels)
                loss.backward() #This is where the model learns by backpropagating
                optimizer.step() #And optimizes its weights
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(self.training_data)))
        print("\nTraining Time (in minutes) =",(time() - start_time) / 60)

    def validation_phase(self, image_to_predict):
        self.model.eval()
        tensor_image = next(iter(self.image_to_tensor(image_to_predict)))
        image = tensor_image[0].view(1, 28, 28)
        with torch.no_grad():
            output = self.model(image) # model returns the vector of raw predictions that a classification model generates.         
        ps = torch.exp(output)
        probab = list(ps.numpy()) # a list of possible numbers
       #  print("Predicted Digit =", probab.index(max(probab)))
        return probab.index(max(probab))
       #####
        
    # Uses the standard MNIST validation data set to test the trained model.
    def mnist_validation(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.validation_data:
                #data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.validation_data.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.validation_data.dataset), 100. * correct / len(self.validation_data.dataset)))


    # Saves the entire trained model to a specific path.
    # @model    trained model
    def save_model(self):
        torch.save(self.model, self.path)
        print('Model is saved')
    
    # Loads entire saved model.
    def load_model(self):
       self.model = torch.load(self.path)
        
    def show(self, img):   # transfer the pytorch tensor(img_tensor) to numpy array
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg), interpolation='nearest')

    # Converts the image which is respresnted as numpy array to a PyTorch readable tensor.
    def image_to_tensor(self, numpy_image):
        return transforms.ToTensor()(numpy_image)