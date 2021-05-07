# Neural network for image processing
</br></br>

## Table of Content
[Introduction](### 1. Introduction)
## Abrivations
ROS......................................................Robot Operationg System
RPC.......................................................Remote Procedcure Call
</br></br></br>

<a name="1. Introduction/>
### 1. Introduction
This documentation was created as part of the project work in the Spezielle Anwendungen der Informatik course in the Applied Computer Science course at HTW Berlin. It describes the implementation of a ROS application, which processes a video stream (the frames as images) and makes use of a fully connected neural network to predict handwritten digits.
</br>
### 1.1.  Project description
The application is distributed over several nodes, with each node taking on a specific task, each node has a specific task. All nodes exchange messages via the publisher subscriber model. The camera node reads the file and sends it to the processor node which does all the preprocessing work. The controller stores the image and the corresponding number. The neural network node predicts the handwritten number depticted on the received image by requesting a trained neural network model.
</br>
### 1.2 ROS introduction
ROS  is an open-source operating system for robots.  It offers a framework, libraries and tools to program the lossperipherals for robots. The communication between the lossely coupled nodes are achieved through the ROS communication infrastructure. Which is based on a publish subscribe message infrastructure, RPC-like services and actions. 
<\br>
### 2. Implementation

#### Roslaunch
roslaunch is a file in .xml format which defines a set of rules how multiple ROS nodes should be launched. 
</br>
### 2.1. Nodes
#### 2.1.1. Camera node
The camera node reads in all images in a folder and publishes them to all subscribing nodes.
The node is launched via ``code`` from the ```roslaunch.xml``` file, the argument specifies the path to the image folder. This allows a dynamic path change without changing the code every time. 
All images are read in and stored as an ```std::unordered_map``` datastrructure which storers data as keys value pairs. The file name corresponds to the key and the associated image is assigned as a value.
``` code example ```
The publish function receives the key, value pairs and publishes the image. to a specific topic and the corrresponding filename to another topic.

#### 2.1.2. Prrocessor node
