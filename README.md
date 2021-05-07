# Neural network for image processing
</br></br>

## Table of Content
- [Abbrivations](#abbrivations)
- [Introduction](#introduction)
- [Project description](#project-description)
- [ROS introduction](#ros-introduction)
- [Implementation](#implementation) 

### Abrivations
[ROS](#ros) .................................................................................Robot Operationg System
[RPC](#rpc) ..................................................................................Remote Procedcure Call
</br>
</br>
</br>
## Introduction
This documentation was created as part of the project work in the Spezielle Anwendungen der Informatik course in the Applied Computer Science course at HTW Berlin. It describes the implementation of a ROS application, which processes a stream of images and makes use of a fully connected neural network as backend to predict handwritten digits on a piece of paper.
</br>
### Project description
The application is distributed over several nodes, with each node taking on a specific task. All nodes exchange messages via the publisher subscriber model. The camera node reads the file and sends it to the processor node which does all the preprocessing work. The controller stores the image and the corresponding number. The neural network node predicts the handwritten number depticted on the received image by requesting a trained neural network model.
</br>
### ROS introduction
ROS  is an open-source operating system for robots.  It offers a framework, libraries and tools to program the lossperipherals for robots. The communication between the lossely coupled nodes are achieved through the ROS communication infrastructure. Which is based on a publish subscribe message infrastructure, RPC-like services and actions. 
</br>
## Implementation

### Nodes

#### roslaunch
roslaunch is a tool which allows to defines a set of rules how multiple ROS nodes should be launched. It basically simplifies the process of launching multiple nodes. Each nodes integrated in the system is defined by a tag and attributes.
```xml 
<launch>
  <node name="ai_service" pgk="imagineer" type="ai_service" />
  ...
</launch>
```
#### Camera node
The camera node reads in all images in a folder and publishes them to all subscribing nodes.</br>
The node is launched via ``code`` from the ```roslaunch.xml``` file, the argument specifies the path to the image folder. This allows a dynamic path change without changing the code every time. All images are read in and stored as an ```std::unordered_map``` datan structure which stores data as key value pairs. The file name corresponds to the key and the associated image is assigned as a value.</br>

```cpp
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











## Sources
