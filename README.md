# Neural network for image processing
</br></br>

## Table of Content
- [Abbrevations](#abbrevations)
- [Introduction](#introduction)
- [Project description](#project-description)
- [ROS introduction](#ros-introduction)
- [Neural network overview](#neural-network-overview)
- [Implementation](#implementation) 
- [roslaunch](#roslaunch)
- [Camera node](#camera-node)
- [Processor node](#processor-node)
- [Controller node](#controller-node)
- [Neural network node](#neural-network-node)

### Abbrevations
[ROS](#ros) .................................................................................Robot Operationg System
[RPC](#rpc) ..................................................................................Remote Procedcure Call
</br>
</br>
</br>
## Introduction
This documentation was created as part of the project work in the Spezielle Anwendungen der Informatik course in the Applied Computer Science course at HTW Berlin. The software application simulates a robot, which processes a stream of images and makes use of a fully connected neural network as backend to predict handwritten digits on a piece of paper.
</br>
### Project description
The application is distributed over several nodes, with each node taking on a specific task. All nodes exchange messages via the publisher subscriber model. The camera node reads the file and sends it to the processor node which does all the preprocessing work. The controller stores the image and the corresponding number. The artificial intelligence node predicts the handwritten number depticted on the received image by requesting a trained neural network model. All nodes are written in C++ except the artifical intelligence node which is written in Python.
</br>

### ROS introduction
ROS  is an open-source operating system for robots. It offers a framework, libraries, tools and to program the different peripherals for robots. The communication between the loosly coupled nodes are achieved through the ROS communication infrastructure. Which is based on a publish subscribe message infrastructure and RPC-like services and actions. 
</br>

### Neural network overview

## Implementation
</br>
### roslaunch
roslaunch is a tool which allows to defines a set of rules how multiple ROS nodes should be launched. It basically simplifies the process of launching multiple nodes. Each nodes integrated in the system is defined by a tag and attributes.
```xml 
<launch>
  <node name="ai_service" pgk="imagineer" type="ai_service" />
  ...
</launch>
```
</br>

### Messages and services
ROS nodes communicate via the known publish subscrber model. Nodes publish content as messages to topis anf other nodes can subscrbe to those topics. </br>
A ROS service is basically a request / reqly model. One node offers a service and another node calls the service by sending a request awaiting a reply.

### Camera node
The camera node reads in all images in a folder and publishes them to all subscribing nodes.</br>
The node is launched via ``code`` from the ```roslaunch.xml``` file, the argument specifies the path to the image folder. This allows a dynamic path change without changing the code every time. All images are read in and stored as an ```std::unordered_map``` datan structure which stores data as key value pairs. The file name corresponds to the key and the associated image is assigned as a value.</br>

```cpp
camera.cpp

for (std::string entry : image_files)
    {
        filename = std::stoi(entry.substr(16, 17));
        cv::Mat image = cv::imread(entry, cv::IMREAD_COLOR);
        message_to_publish[filename] = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg(); // adds filename as key and cv_bridge Image as value  
    }
    return message_to_publish; 
```

The publish function receives the key, value pairs and publishes the color image to a specific topic and the corresponding filename to another topic.
</br>

#### Processor node
The processor node performs some manipulations on the photo that are necessary for further processing.</br>
After sucessfully initializing the node via the roslaunch file, the subscriber function is called and subscribes to the image topic. If an image is received, the corresponding callback function is called.</br>

```cpp
subscriber = transport.subscribe("camera/image", 1, &Processor::callback, this);
```
The image processing function call converts the received ROS image message to a manipulable OpenCV image format. In order to save space and publish the image more efficiently, the image is reduced by half and returned as OpenCV image back to the callback function.   
```c++
void callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", resized_image).toImageMsg()); 
        ROS_INFO("Image is published from processor node.");
    }
```      
Finally, the photo is converted back to the ROS Image message format and published as grayscale to the controller node.

### Controller node
The controller node subscribes to all two topics, stores them and publishes the image to the neural network node.
After the node has been initialized, the controller object subscribes to the number topic published by the camera node and the topic set up by the processor node. Both topics get synchronized based on their set time stamp at publishing time. 

```c++
controller.cpp

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
The service for requesting the predicted number is also initialized in the constructor of the controller class. To save the digit and corresponding image in a ```std::vector``` data structure both are copied to a new object NumberAndPicture, which acts as the data type of the vector.
```c++
storage.push_back(NumberAndPicture(digit, saved_image));
```
Once the object has been saved, the image is sent as a service to the artificial intelligence node and the callback awaits the response from the requested service node.
## Sources
