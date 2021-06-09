#!/usr/bin/env python
import rospy, torch, pathlib
from sensor_msgs.msg import Image
from imagineer.srv import ImageAck, ImageAckResponse
from ai_service.ai_service import AiService

# Function is called if the node receives a messages via the subscribed topic.
# @request    the received image. 
def callback(request, arg):
    response = ImageAckResponse()
    response.result = 5 ## later the predicted number is passed to response.result
    return response

# Handles all the basics like initializing node, receiving images through cv_bridge 
# and checking if Nvidias cuda is available
def main():
    rospy.init_node('ai_service')
    save_path = '/home/marta/catkin_ws/src/imagineer/my_trained_mnist_model.pt'
    service = AiService(save_path)
    file = pathlib.Path(save_path)
    if not file.exists():
        print('No model found. Training in progrress')
        service.training_phase()
        service.save_model()
    else:
        service.load_model()
        print('Model found and loaded. Validation in progress')
        service.validation_phase()
        rospy.Service('image_ack', ImageAck, callback, service)
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()