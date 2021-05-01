from __future__ import print_function
import rospy, cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from image_recognizer.srv import ImageAck, ImageAckResponse

# Function is called if the node receives a messages via the subscribed topic.
# @image    the received image. 
def callback(request):
    print('Got image')
    number = '1'
    return ImageAckResponse(number))

# Function to handle all the basics like initializing node, receiving images through cv_bridge.
def main():
    rospy.init_node('ai_service')
    rospy.loginfo('Neural network node started')
    #rospy.Subscriber('saved/image', Image, callback)
    service = rospy.Service('image_ack', ImageAck, callback)
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == 'main':
    try:
        main()
    except rospy.ROSInterruptException as error:
        print('Neural network node could not be started: ', error)