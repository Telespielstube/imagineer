#!/usr/bin/env python
from __future__ import print_function

import rospy, cv2, torch, os, platform
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from imagineer.srv import ImageAck, ImageAckResponse
from number_machine import NumberMachine
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose

class NumberMachine():

    def __init__(self):
        print('running')
        super().__init__()
        # self.flatten = nn.Flatten()
    
    def send_ok(self):
        ok = 5
        return ok 

# Function is called if the node receives a messages via the subscribed topic.
# @image    the received image. 
def callback(request, args):
    response = ImageAckResponse()
    num_machine = args[0]
    response.result = num_machine.send_ok()
    return response

# Handles all the basics like initializing node, receiving images through cv_bridge, initializing pytorch datasaets 
# for trainig and test environment.
def main():
    rospy.init_node('ai_service')
    rospy.loginfo('Neural network node is running')
    
    num_machine = NumberMachine()
    rospy.Service('image_ack', ImageAck, callback, (num_machine))
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()