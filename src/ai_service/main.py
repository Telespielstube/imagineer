#!/usr/bin/env python
import rospy, torch, pathlib, sys
from sensor_msgs.msg import Image
from imagineer.srv import ImageAck, ImageAckResponse
from ai_service.ai_service import AiService
from ai_service.storage import Storage

# Function is called if the node receives a messages via the subscribed topic.
# @request    the received image as sensor message. 
def callback(request, service):
    response = ImageAckResponse() 
    response.result = service.prediction(request.image)
    return response
     
# Handles all the basics like initializing node, ai_service and the Service server. Checks if a model is already saved 
# or loads a stored model. 
def main():
    rospy.init_node('ai_service')
    print('Service is running.')
    
    file_name = pathlib.Path(sys.argv[1])
    print(f'{file_name}')
    ai_service = AiService()
    storage = Storage(sys.argv[1], ai_service)
    if not file_name.exists():
        print('No model found. Training in progress')
        ai_service.training()
        ai_service.validating_mnist()
        storage.save_model()
    else:
        storage.load_model()
        print('Model found.')
        ai_service.validating_mnist()
        rospy.Service('image_ack', ImageAck, lambda request : callback (request, ai_service))
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()
