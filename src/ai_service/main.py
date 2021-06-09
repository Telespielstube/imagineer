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
    service = AIService(save_path)
    file = pathlib.Path(save_path)
    if not file.exists():
        print('Does not exist')
        # checks if Nvidia cuda support is available. 
        if torch.cuda.is_available():
            torch.device('gpu')
            print('Using the gpu')
            service.training_phase_with_cuda()
            service.save_model()
        else:
            torch.device('cpu')
            print('Using the cpu. Get youself a cup of coffee it will take time :-D')
            service.training_phase_without_cuda()
            service.save_model()
    else:
        print('Model does exist')
        service.load_model()
        service.validation_phase()
        rospy.Service('image_ack', ImageAck, callback, service)
        rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()