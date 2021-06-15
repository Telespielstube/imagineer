import rospy, torch 
import numpy as np
import matplotlib.pyplot as plt
from time import time
from ai_service.neural_network import NeuralNetwork
from torch import nn
from cv_bridge import CvBridge
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose
class AiService():

    def __init__(self, save_path):
        self.batch_size = 2800
        self.epochs = 25
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.training_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
        self.validation_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
        self.path = save_path
        self.cv_bridge = CvBridge()
        self.model = NeuralNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Function to train the mnist dataset.
    def training(self):
        criterion = nn.CrossEntropyLoss() #combines LogSoftmax and NLLLoss in one single class.
        optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate, self.momentum)
        start_time = time()
        for epoch in range(self.epochs):
            running_loss = 0
            # trainig phase
            for images, labels in self.training_data:
                optimizer.zero_grad() 
                image, label = images.to(self.device), labels.to(self.device)
                output = self.model(image)
                loss = criterion(output, label)
                loss.backward() #This is where the model learns by backpropagating
                optimizer.step() #optimizing weights
                running_loss += loss.item() # Returns the value of this tensor as a standard Python number
            else:
                print("Epoch {} - Training loss: {:.10f}".format(epoch, running_loss / len(self.training_data)))
        # elapsed_time = (time.time() - start_time) / 60
        # print("\nTraining Time (in minutes): {} = ", (time() - ti)/60)

    # Function validates the trained model against the received image.
    # @request_image    image object to be validated.
    # @return           the predicted number. 
    def validating(self, request_image):
        self.model.eval()
        tensor_image = self.image_to_tensor(request_image) 
        normalized_image = self.normalize_image(tensor_image)  
        with torch.no_grad():
            output = self.model(normalized_image) # model returns the vector of raw predictions that a classification model generates.         
        probability = output.cpu().data.numpy().argmax() #moves tensor to cpu and converts it to numpy array
        rospy.loginfo('Output: %s', probability)      
        return probability #return the number with the largest predicted probability.
    
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

    # Normalizes the tensor_image so every image is aligned correctly.
    # @tensor_image    image object in PyTorrch tensorr format.
    #
    # @return          correctly aligned image.
    def normalize_image(self, tensor_image):
        normalize = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        return normalize(tensor_image)

    # Converts the ROS sensor rmessage image to a PyTorch readable tensor.
    # @requsted_image    the image still in ROS sensor message format.
    #
    # @return      ROS sensor message format converted to PyTorch tensor.
    def image_to_tensor(self, request_image):
        img = self.cv_bridge.imgmsg_to_cv2(request_image, 'mono8')
        plt.imshow(img, cmap='Greys')
        return transforms.ToTensor()(img) #(self.cv_bridge.imgmsg_to_cv2(request_image, 'mono8'))