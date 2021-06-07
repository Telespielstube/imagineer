from __future__ import print_function

import rospy, cv2, torch, os, platform
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from imagineer.srv import ImageAck, ImageAckResponse
from ai_service.service import Service


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
    service = Service()
    if not service.load_model():
        service.training_phase()
        service.save_model()
    else:
        rospy.Service('image_ack', ImageAck, callback)
        rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()