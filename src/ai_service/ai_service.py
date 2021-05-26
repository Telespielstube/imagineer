#!/usr/bin/env python

from __future__ import print_function

import rospy, cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from imagineer.srv import ImageAck, ImageAckResponse

# Function is called if the node receives a messages via the subscribed topic.
# @image    the received image. 
def callback(request):
    print('Got image')
    number = 2
    request.result = number
    return ImageAckResponse(request)

# Handles all the basics like initializing node, receiving images through cv_bridge, initializing pytorch datasaets 
# for trainig and test environment.
def main():
    rospy.init_node('ai_service')
    rospy.loginfo('Neural network node is running')
    #wait the service to be advertised, otherwise the service use will fail
    rospy.Service('image_ack', ImageAck, callback) #(train_dataloader, test_dataloader))
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()