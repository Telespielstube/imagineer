import torch
from time import time
from ai_service.neural_network import NeuralNetwork
from torch import nn
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
        self.model = NeuralNetwork()
        self.path = save_path

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

    def validation_phase(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        min_valid_loss = 0.0
        for images, labels in self.validation_data:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
        
            target = self.model(data)
            loss = criterion(target,labels)
            valid_loss = loss.item() * data.size(0)

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
            # img = images[0].view(1, 784)
            # with torch.no_grad():
            #     logps = self.model(img)
            # ps = torch.exp(logps)
            # probab = list(ps.numpy()[0])
            # print("Predicted Digit =", probab.index(max(probab)))

    # Saves the entire trained model to a specific path.
    # @model    trained model
    def save_model(self):
        torch.save(self.model, self.path)
        print('Model is saved')
    
    # Loads entire saved model.
    def load_model(self):
        return torch.load(self.path)
        

    #model.set_image(request.image)
    #model.training_phase(model) 

    # Sets the image property attribute.
    # @image        image sent from the controller node. 
    # def set_image(self, image):
    #     self.image = image

    # def image_to_tensor(self):
    #     image_to_numpy = numpy.asarray(self.image)
    #     return transforms.ToTensor()(image_to_numpy)