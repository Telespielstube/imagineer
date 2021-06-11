#!/usr/bin/env python
import rospy, torch, pathlib, numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from imagineer.srv import ImageAck, ImageAckResponse
from ai_service.ai_service import AiService


# Function is called if the node receives a messages via the subscribed topic.
# @request    the received image as sensor message. 
def callback(request, service):
    response = ImageAckResponse()
    cv_bridge = CvBridge()
    cv_image = convert_to_cv_image(cv_bridge, request.image)
    response.result = service.validation_phase(cv_image) 
    return response

# Converts the ROS sensor message to a PyTorch compatible numpy array. 
# @cv_bridge      CvBridge object needed to convert from sensor message to numpy
# @image          The image sent from the controller node.
# 
# @return         numpy array which represents the image. 
def convert_to_cv_image(cv_bridge, image):
    return cv_bridge.imgmsg_to_cv2(image, 'mono8')
   
# Handles all the basics like initializing node, ai_service and the Service server. Checks if a model is already saved 
# or loads a stored model. 
def main():
    rospy.init_node('ai_service')
    save_path = '/home/marta/catkin_ws/src/imagineer/my_trained_mnist_model.pt'
    ai_service = AiService(save_path)
    file = pathlib.Path(save_path)
    if not file.exists():
        print('No model found. Training in progress')
        ai_service.training_phase()
        ai_service.mnist_validation()
        ai_service.save_model()
    else:
        ai_service.load_model()
        print('Model found and loaded. Validation in progress')
        ai_service.mnist_validation()
        rospy.Service('image_ack', ImageAck, lambda request : callback (request, ai_service))

    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()