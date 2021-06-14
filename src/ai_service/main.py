#!/usr/bin/env python
import rospy, torch, pathlib
from sensor_msgs.msg import Image
from imagineer.srv import ImageAck, ImageAckResponse
from ai_service.ai_service import AiService

# Function is called if the node receives a messages via the subscribed topic.
# @request    the received image as sensor message. 
def callback(request, service):
    response = ImageAckResponse() 
    response.result = service.validation_phase(request.image)
    return response
     
# Handles all the basics like initializing node, ai_service and the Service server. Checks if a model is already saved 
# or loads a stored model. 
def main():
    rospy.init_node('ai_service')
    save_path = '/home/marta/catkin_ws/src/imagineer/my_trained_mnist_model.pt'
    ai_service = AiService(save_path)
    file_name = pathlib.Path(save_path)
    if not file_name.exists():
        print('No model found. Training in progress')
        ai_service.training_phase()
        ai_service.mnist_validation()
        ai_service.save_model()
    else:
        ai_service.load_model()
        print('Model found and loaded.')
        ai_service.mnist_validation()
        rospy.Service('image_ack', ImageAck, lambda request : callback (request, ai_service))
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()