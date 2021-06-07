#!/usr/bin/env python
from __future__ import print_function

import rospy, cv2, torch, os, platform
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from imagineer.srv import ImageAck, ImageAckResponse
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
    #model = arg
    #print(model)
    #model.set_image(request.image)
    #model.training_phase() 
    response.result = 5 ## later the predicted number is passed to response.result
    return response

# Handles all the basics like initializing node, receiving images through cv_bridge, initializing pytorch datasaets 
# for trainig and test environment.
def main():
    rospy.init_node('ai_service')
    model = NumberMachine(batch_size=200, epochs=10, learning_rate=0.01, log_interval=10)
    rospy.Service('image_ack', ImageAck, callback)
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()