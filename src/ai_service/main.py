import rospy
from sensor_msgs.msg import Image
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
    service.training_phase()
    service.save_model()

    rospy.Service('image_ack', ImageAck, callback)
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == '__main__':
    main()