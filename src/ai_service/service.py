import torch
from time import time
from ai_service.neural_network import NeuralNetwork
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose

class Service():

    def __init__(self):
        self.batch_size = 200
        self.epochs = 5
        self.learning_rate = 0.01
        self.training_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, 
                                    transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
        self.validation_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
        self.model = NeuralNetwork()

    #model.set_image(request.image)
    #model.training_phase(model) 
    def training_phase_without_cuda(self):
        print("Training is running")
        criterion = nn.CrossEntropyLoss() #combines LogSoftmax and NLLLoss in one single class.
        optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        start_time = time()
        for epoch in range(self.epochs):
            running_loss = 0
            for images, labels in self.training_data:
                images = images.view(images.shape[0], -1) # Flatten MNIST images into a 784 long vector
                optimizer.zero_grad() # Training pass         
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward() #This is where the model learns by backpropagating
                optimizer.step() #And optimizes its weights
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(self.training_data)))
        print("\nTraining Time (in minutes) =",(time() - start_time) / 60)

    def training_phase_with_cuda(self):
        print("Training is running")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        time0 = time()
        for epoch in range(10):
            running_loss = 0
            for images, labels in self.training_data:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)           
                # Training pass
                optimizer.zero_grad()               
                output = self.model(images.cuda)
                loss = criterion(output, labels.cuda)                
                #This is where the model learns by backpropagating
                loss.backward()              
                #And optimizes its weights here
                optimizer.step()
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(self.training_data)))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)

    # Saves the entire trained model to a specific path.
    # @model    trained model
    def save_model(self):
        torch.save(self.model, './my_trained_mnist_model.pt')
        print('Model is saved')
    
    # Loads entire saved model.
    def load_model(self):
        self.model = torch.load('./my_trained_mnist_model.pt')
        return self.model.eval()


      # Sets the image property attribute.
    # @image        image sent from the controller node. 
    # def set_image(self, image):
    #     self.image = image

    # def image_to_tensor(self):
    #     image_to_numpy = numpy.asarray(self.image)
    #     return transforms.ToTensor()(image_to_numpy)