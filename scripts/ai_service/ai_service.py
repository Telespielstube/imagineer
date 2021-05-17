from __future__ import print_function
import rospy, cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from imagineer.srv import ImageAck, ImageAckResponse
from scripts.ai_service.number_cruncher import NumberCruncher

# Function is called if the node receives a messages via the subscribed topic.
# @image    the received image. 
def callback(request):
    print('Got image')
    NumberCruncher(request)
    number = '2'
    return ImageAckResponse(number)

# Function to handle all the basics like initializing node, receiving images through cv_bridge.
def main():
    rospy.init_node('ai_service')
    rospy.loginfo('Neural network node started')
    rospy.Service('image_ack', ImageAck, callback)
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == 'main':
    main()