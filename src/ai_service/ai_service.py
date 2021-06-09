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
        self.learning_rate = 0.001
        self.training_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
        self.validation_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
        self.path = save_path
        self.model = NeuralNetwork()
        if torch.cuda.is_available():
            print('Cuda is supported')
            self.model = self.model.cuda()
        
    # Function to train the mnist dataset.
    def training_phase(self):
        print("Training phase")
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
        #image = image_to_predict.view(1, 784)
        with torch.no_grad():
            logps = self.model(image_to_predict) # model returns the vector of raw predictions that a classification model generates.         
        ps = torch.exp(logps)
        probab = list(ps.numpy()) # a list of possible numbers
        print("Predicted Digit =", probab.index(max(probab)))
        self.show(image_to_predict)

    # Saves the entire trained model to a specific path.
    # @model    trained model
    def save_model(self):
        torch.save(self.model, self.path)
        print('Model is saved')
    
    # Loads entire saved model.
    def load_model(self):
        return torch.load(self.path)
        
    def show(self, img):   # transfer the pytorch tensor(img_tensor) to numpy array
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg), interpolation='nearest')

    # def image_to_tensor(self):
    #     image_to_numpy = numpy.asarray(self.image)
    #     return transforms.ToTensor()(image_to_numpy)