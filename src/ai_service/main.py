#!/usr/bin/env python
import rospy, torch, pathlib, numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from imagineer.srv import ImageAck, ImageAckResponse
from ai_service.ai_service import AiService


# Function is called if the node receives a messages via the subscribed topic.
# @request    the received image. 
def callback(request, arg):
    response = ImageAckResponse()
    service = arg[0]
    cv_bridge = CvBridge()
    numpy_image = convert_to_numpy_image(cv_bridge, request.image)
    service.validation_phase(numpy_image)
    response.result = 5 ## later the predicted number is passed to response.result
    return response

# Converts the ROS sensor message to a PyTorch compatible numpy array. 
# @cv_bridge      CvBridge object needed to convert from sensor message to numpy
# @image          The image sent from the controller node.
# 
# @return         numpy array which represents the image. 
def convert_to_numpy_image(cv_bridge, image):
    cv_image = cv_bridge.imgmsg_to_cv2(image, 'mono8')
    return numpy.asarray(cv_image)

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
        rospy.Service('image_ack', ImageAck, callback, service)
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()