#!/usr/bin/env python
from __future__ import print_function

import rospy, cv2, torch, os, platform
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from imagineer.srv import ImageAck, ImageAckResponse
from time import time
from ai_service.number_machine import NumberMachine
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose

# Function is called if the node receives a messages via the subscribed topic.
# @request    the received image. 
# @args       arguments passed to callback function.
def callback(request):
    response = ImageAckResponse()
    response.result = 5 ## later the predicted number is passed to response.result
    return response

# Handles all the basics like initializing node, receiving images through cv_bridge, initializing pytorch datasaets 
# for trainig and test environment.
def main():
    rospy.init_node('ai_service')
    training_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
    validation_data = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=False, download=True, 
                            transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])), 200, shuffle=True)
    model = NumberMachine(batch_size=200, epochs=10, learning_rate=0.01, log_interval=10)
    #model.set_image(request.image)
    #model.training_phase(model) 
    print("Training is running")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    time0 = time()
    for epoch in range(10):
        running_loss = 0
        for images, labels in training_data:
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
            print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(training_data)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)


    rospy.Service('image_ack', ImageAck, callback)
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()