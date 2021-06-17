# Neural network for image processing
</br></br>

## Table of Content
- [Abbrevations](#abbrevations)
- [Introduction](#introduction)
- [Project description](#project-description)
- [ROS overview](#ros-overview)
- [Neural network overview](#neural-network-overview)
- [Implementation](#implementation) 
- [roslaunch](#roslaunch)
- [Nodes](#nodes)
- [Camera node](#camera-node)
- [Processor node](#processor-node)
- [Controller node](#controller-node)
- [Neural network node](#neural-network-node)

### Abbrevations
[ROS](#ros) .................................................................................Robot Operationg System</br>
[RPC](#rpc) ...................................................................................Remote Procedcure Call
</br>
</br>
</br>
## Introduction
This documentation was created as part of the project work in the Spezielle Anwendungen der Informatik course in the Applied Computer Science course at HTW Berlin. 
</br>
### Project description
The software simulates a robot application, which processes a stream of images and uses a fully connected neural network as backend to predict handwritten digits on a piece of paper. The application is distributed over several nodes, with each node taking on a specific task. All nodes exchange messages via the publisher subscriber model. The camera node reads the file and sends it to the processor node which does all the preprocessing work. The controller stores the image and the corresponding number. The artificial intelligence node predicts the handwritten number depticted on the received image by requesting a trained neural network model. All nodes are written in C++(1) except the artifical intelligence node which is written in Python(2).
</br>

### ROS overview
ROS(3)  is an open-source operating system for robots. It offers a framework, libraries, tools and to program the different peripherals for robots. The communication between the loosly coupled nodes are achieved through the ROS communication infrastructure. Which is based on a publish subscribe message infrastructure and RPC-like services and actions. 
</br>
### Neural network overview
In simple terms, a neural network mimics a human brain. Just like a human brain, the artificial neural network links nodes using weighted edges. This means that the network can be trained through several training runs and thus predict results. By modifying the weighted edges the system improve the learning rate and prediction results.

## Implementation

### Messages and services
Each ROS node only performs one specific task. Therefore the nodes need to communicate in some way. ROS nodes communicate via the known publish subscrber model. Nodes publish content as messages to topis and other nodes can subscrbe to those topics. </br> The advantage of this model is an instant notification about news, parallelism and scalability.
A ROS service is basically a request / reqly model. One node offers a service and another node calls the service by sending a request awaiting a reply. The advantage of this model is an instant notification about news, parallelism and scalability.

### roslaunch
roslaunch is a tool which allows to define a set of rules how multiple ROS nodes should be launched. It basically simplifies the process of launching multiple distributed nodes. Each nodes integrated in the system is defined by a tag containig some attributes. The nodes which are launched via arguments are the camera node and the neural network node. The camera gets the path to the images passed by argument and the neural network gets the path where to save the trained model by argument.
In order to launch each node with ```roslaunch``` only one command is necessary now.</br>
```roslaunch imagineer startup.launch```

## Nodes
Each node perfoms a specific task in the image recognizing workflow which is laid out in the following section.
### Camera node
The node is launched with an additional argument which specifies the path to the image folder. All images are read in and stored as ```std::vector``` entries. After all files are stored a random number is calculated from the number range of the vector size. If a number is calculated the file on the specific position is picked the ```std::string``` is converted to a OpenCV image format and the corresponding number is sliced from the path at the exact position. 
```cpp
int filename = std::stoi(image_file.substr(16, 17));
cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
```
Both variables are passed to the image object. The publish function puts the object attributes in two different topics an sends them to all available subscribers.

### Processor node
The processor node performs some manipulations on the photo that are necessary for further processing.</br>
After sucessfully initializing the node via the roslaunch file, the subscriber function is called and subscribes to the image topic. If an image is received, the corresponding callback function is called.</br>

```c++
subscriber = transport.subscribe("camera/image", 1, &Processor::callback, this);
```
The image processing function call converts the received ROS image message to a manipulable OpenCV image format. In order to save space and publish the image more efficiently, the image is reduced by half and returned as OpenCV image back to the callback function.   
```c++
void callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); 
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", resized_image).toImageMsg()); 
    }
```      
At last, the photo is converted back to a ROS Image message format and gets published as grayscale image to the controller node.

### Controller node
After the node has been initialized, the controller object subscribes to the number topic published by the camera node and the topic set up by the processor node. Both topics get synchronized based on their set time stamp at publishing time. 

```c++
Controller() {
    img_subscriber.subscribe(node, "processor/image", 1);
    int_subscriber.subscribe(node, "camera/integer", 1); 
    sync.reset(new message_filters::TimeSynchronizer<sensor_msgs::Image, imagineer::Number>(img_subscriber, int_subscriber, 10));
    ...
}
```
After both messages are received they get syncronized by their time stamps in their headers. The ```TimeSynchronizer``` function channels both messages into one callback. To achieve the bundling of topics, the TimeSynchronizer is declared as a member variable wrapped with a shared pointer, which allows sharing of the pointed object.
```c++
boost::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::Image, imagineer::Number>> sync;
```
The service for requesting the predicted number is also initialized in the constructor of the controller class. To save the digit and corresponding image in a ```std::vector``` data structure both are copied to a new object named NumberAndPicture, which acts as the data type of the vector. Once the object has been saved, the image is sent as a service message to the artificial intelligence node and the callback blocks until the response from the requested service node arrives.

### Neuronal network node
The neural network node consists of two parts the service and the underlying neural network which is responsible for the image regogniction.
Before the actual image recognition process, the neural network must be trained first by using the MNIST(2) datasets. The neural network is built up with three hidden layers. The input layer contains 784 neurons, each neuron stands for one pixel of the image to be recognized. The 3 hidden layers reduce the number of neurons gradually, up to the output layer which contains 10 neurons for the classification of the predicted number. Once the network is initialized the next step is to train it.
The training function creates an optimizer object with the SGD algorithm. It stands for stochastic gradient descent and means that 



The incomming service message contains the image as a ROS sensor message. The callback function is wrapped in a lambda function which allows to take the service object as additional argument.
```python
rospy.Service('image_ack', ImageAck, lambda request : callback (request, ai_service))
```
In order to use the ROS sensor message image in the neural network properly it needs to be converted to PyTorch's Tensor format.
```python
def image_to_tensor(self, request_image):
        return transforms.ToTensor()(cv_bridge.imgmsg_to_cv2(request_image, 'mono8'))
```


## Sources
(MNIST)[http://yann.lecun.com/exdb/mnist/]
(C++)[https://www.cplusplus.com]