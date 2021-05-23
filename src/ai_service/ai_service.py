from __future__ import print_function
import rospy, cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from imagineer.srv import ImageAck, ImageAckResponse
#from imagineer.number_cruncher import NumberCruncher
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

class NumberCruncher:
    
    # Constructor
    # @self    this object every object function has to have self.
        def __init__(self):
            super(NumberCruncher, self).__init__()
            self.flatten = nn.Flatten()

# Function is called if the node receives a messages via the subscribed topic.
# @image    the received image. 
def callback(request, args):
    print('Got image')
   # NumberCruncher(request, args[0], args[1])
    number = '2'
    return ImageAckResponse(number)

# Function to handle all the basics like initializing node, receiving images through cv_bridge, initializing pytorch datasaets 
# for trainig and test environment.
def main():
    rospy.init_node('ai_service')
    rospy.loginfo('Neural network node is running')
    batch_size = 64
    mnist_training_data = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=None)
    mnist_test_data = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=None)
    train_dataloader = DataLoader(mnist_training_data, batch_size=batch_size)
    test_dataloader = DataLoader(mnist_test_data, batch_size=batch_size)
    rospy.Service('image_ack', ImageAck, callback, (train_dataloader, test_dataloader))
    rospy.spin()

# Implies that the script is run standalone and cannot be imported as a module.
if __name__ == 'main':
    main()