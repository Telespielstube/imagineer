#!/usr/bin/env python
import torch, os, numpy
from torch import nn
from cv_bridge import CvBridge
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose
from ai_service.neural_network import NeuralNetwork

class AiService():

    def __init__(self, save_path):
        self.batch_size = 256
        self.epochs = 15
        self.learning_rate = 0.001
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.training_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=self.transform), batch_size=self.batch_size, shuffle=True)
        self.validation_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=self.transform), batch_size=self.batch_size, shuffle=True)
        self.path = save_path
        self.cv_bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork()
        self.model.to(self.device)
        
    # Function to train the mnist dataset.
    def training(self):
        criterion = nn.CrossEntropyLoss() #combines LogSoftmax and NLLLoss in one single class.
        optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        for epoch in range(self.epochs):
            running_loss = 0
            # trainig phase
            for images, labels in self.training_data:
                optimizer.zero_grad() 
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward() 
                optimizer.step() 
                running_loss += loss.item() # the value of this tensor as a standard Python number
                print("Epoch {} - Training loss: {:.10f}".format(epoch, running_loss / len(self.training_data)))

    # Function validates the trained model against the received image.
    # @request_image    image object to be validated.
    # @return           the predicted number. 
    def prediction(self, request_image):
        self.model.eval()
        normalized_tensor = self.image_to_normalized_tensor(request_image)   
        with torch.no_grad():
            output = self.model(normalized_tensor)
        return output.cpu().data.numpy().argmax() #moves tensor to cpu and converts it to numpy array and returns the number with the largest predicted probability.
    
    # Uses the standard MNIST validation data set to test the trained model.
    def validating_mnist(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for image, label in self.validation_data:
                image, label = image.to(self.device), label.to(self.device)
                output = self.model(image)
                test_loss += criterion(output, label).item()  # sums up batch loss
                pred = output.argmax()
                correct += pred.eq(label.view_as(pred)).sum().item() #sums up the correct predicted numbers
        test_loss /= len(self.validation_data.dataset)
        print('\n Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.validation_data.dataset), 100. * correct / len(self.validation_data.dataset)))

    # Saves the entire trained model to a specific path.
    def save_model(self):
        save_folder = '/home/marta/catkin_ws/src/imagineer/saved_models/'
        try:
            os.mkdir(save_folder)
        except FileExistsError:
            pass
        torch.save(self.model, self.path)
        print('Model is saved')
    
    # Loads entire saved model.
    def load_model(self):
       self.model = torch.load(self.path)

    # Converts the ROS sensor message to a PyTorch tensor and normalizes the tensor_image 
    # that every image is aligned correctly.
    # @tensor_image    image object in PyTorrch tensorr format.
    #
    # @return          correctly aligned image in PyTorch's tensor format
    def image_to_normalized_tensor(self, tensor_image):
        tensor = transforms.ToTensor()(self.cv_bridge.imgmsg_to_cv2(tensor_image, 'mono8'))
        normalize = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        return normalize(tensor)