import rospy
import torch
import pathlib
from sensor_msgs.msg import Image
from imagineer.srv import ImageAck, ImageAckResponse
from ai_service.service import Service

# Function is called if the node receives a messages via the subscribed topic.
# @request    the received image. 
def callback(request):
    response = ImageAckResponse()
    response.result = 5 ## later the predicted number is passed to response.result
    return response

# Handles all the basics like initializing node, receiving images through cv_bridge 
# and checking if Nvidias cuda is available
def main():
    rospy.init_node('ai_service')
    service = Service()
    file = pathlib.Path('./src/imagineer/my_trained_mnist_model.pt')
    if not file.exists():
        print('Does not exist')
        # checks if Nvidia cuda support is available. 
        if torch.cuda.is_available():
            torch.device('gpu')
            print('Using the gpu')
            service.training_phase_with_cuda()
            service.save_model()
            return
        else:
            torch.device('cpu')
            print('Using the cpu. Get youself a cup of coffee it will take time :-D')
            service.training_phase_without_cuda()
            service.save_model()
    else:
        print('does exist')
        service.load_model()
        rospy.Service('image_ack', ImageAck, callback)
        rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()