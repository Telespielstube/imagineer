#!/usr/bin/env python
from __future__ import print_function

import rospy, cv2, torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#from number_machine import NumberMachine
from imagineer.srv import ImageAck, ImageAckResponse
#from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose


# Function is called if the node receives a messages via the subscribed topic.
# @image    the received image. 
def callback(request, args):
    response = ImageAckResponse()
    print('Got image')
    ok = 1
    response.result = ok
    return response

# Handles all the basics like initializing node, receiving images through cv_bridge, initializing pytorch datasaets 
# for trainig and test environment.
def main():
    rospy.init_node('ai_service')
    rospy.loginfo('Neural network node is running')
   # num_machine = NumberMachine()
   # training_data = datasets.MNIST(root='./data', train=True, download=True, transform=None)
   # test_data = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    rospy.Service('image_ack', ImageAck, callback) # training_data, test_data)
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()