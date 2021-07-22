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
- [Graph](#graph)
- [Conclusion](#conclusion)
- [Sources](#sources)
- [Figures](#figures)

### Abbrevations
[ROS](#ros)   ...............................................................................Robot Operationg System</br>
[RPC](#rpc)   ..................................................................................Remote Procedcure Call</br>
[SGD](#sgd)   ...........................................................................Stochastic gradient descent</br>
[Adam](#adam) ..........................................................................Adaptive Moment Estimation</br>
</br>
</br>
</br>
## Introduction
This documentation was created as part of the project work in the Spezielle Anwendungen der Informatik course in the Applied Computer Science course at HTW Berlin. 
</br>
### Project description
The software simulates a robot application, which processes a stream of images and uses a fully connected neural network as backend to predict handwritten digits on a piece of paper. The application is distributed over several nodes, with each node taking on a specific task. All nodes exchange messages via the publisher subscriber model. The camera node reads the file and sends it to the processor node which does all the preprocessing work. The controller stores the image and the corresponding number. The artificial intelligence node predicts the handwritten number on the received image using a trained neural network model. All nodes are written in C++(1) except the artifical intelligence node which is written in Python(2).
</br>
### ROS overview
ROS(3) is an open-source operating system for robots. It offers a framework, libraries, tools and to program the different peripherals for robots. The communication between the loosly coupled nodes is achieved through the ROS communication infrastructure, which is based on a publish subscribe message infrastructure and RPC-like services and actions. 
</br>
### Neural network overview
A neural network mimics a human brain. Just like a human brain, the artificial neural network links nodes using weighted edges. This means that the network can be trained through several training runs and thus predict results. By modifying the weighted edges the system improve the learning rate and prediction results. Especially the neural network with its use of the PyTorch(4) framework makes it concise and easier to understand the complexity behind it.</br>
</br>
## Implementation
### Messages and services
Each ROS(3) node performs only one specific task. Therefore the nodes need to communicate in some way. ROS nodes communicate via the well known publish subscriber model(5). Nodes publish content as messages to topics and other nodes can subscribe to those topics. </br> The advantage of this model is an instant notification about news, parallelism and scalability.
A ROS service(6) is basically a request / reply model. One node offers a service and another node calls the service by sending a request awaiting a reply. The advantage of this model is an instant notification about news, parallelism and scalability.

### roslaunch
roslaunch(7) is a tool which allows to define a set of rules how multiple ROS nodes should be launched. It basically simplifies the process of launching multiple distributed nodes. Each nodes integrated in the system is defined by a tag containig some attributes. The nodes which are launched via arguments are the camera node and the neural network node. 
In order to launch each node with ```roslaunch``` only one command is necessary now.</br>
```roslaunch imagineer startup.launch```

## Nodes
Each node perfoms a specific task in the image recognizing workflow which is laid out in the following section.
### Camera node
The node is launched with an additional argument which specifies the path to the image folder. All images are read in and stored as ```std::vector``` entries. After all files are stored, a random number is calculated from the number range of the vector size. Each number the random number function returns picks the file on the specific position from the file list. Then the file content gets converted to a OpenCV image format and the file name string is converted to an integer. 
Finally, the publish function puts the image object and integer variable in two different topics and sends them to all available subscribers.

### Processor node
The processor node performs some manipulations on the photo that are necessary for further processing.</br>
After sucessfully initializing the node via the roslaunch file, the subscriber function is called and subscribes to the image topic. If an image is received, the corresponding callback function is called.</br>

```c++
subscriber = transport.subscribe("camera/image", 1, &Processor::callback, this);
```
The image processing function call converts the received ROS image message to a manipulable OpenCV image format. In order to adapt the images to the size of the MNIST(9) images, they are scaled down to 28 x 28 pixels. Furthermore, the image is converted to grayscale,  inverted and a certain threshold is applied. This is neccessary for better edge detection respectively object detection in the image. 
The returned image is converted back to the ROS sensor message format and gets sent to the controller node.
```c++
void callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::Mat processed_image = process_image(cv_bridge::toCvCopy(message)->image); 
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", processed_image).toImageMsg()); 
    }
```      

### Controller node
After the node has been initialized, the controller node subscribes to the number topic published by the camera node and the new image topic set up by the processor node.
```c++
Controller() {
    img_subscriber.subscribe(node, "processor/image", 1);
    int_subscriber.subscribe(node, "camera/integer", 1); 
    sync.reset(new message_filters::TimeSynchronizer<sensor_msgs::Image, imagineer::Number>(img_subscriber, int_subscriber, 10));
    ...
}
```
Once both messages are received they get syncronized by their time stamps in their headers. The ```TimeSynchronizer``` function channels both messages into one callback. To achieve the bundling of topics, the TimeSynchronizer is declared as a member variable wrapped with a shared pointer, which allows accessing the object it points to in a memory-safe way.
```c++
boost::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::Image, imagineer::Number>> sync;
```
The service(6) for requesting the predicted number is also initialized in the constructor of the controller class. To save the digit and corresponding image in a ```std::vector``` data structure, both are copied to a new object named NumberAndPicture, which acts as the data type of the vector. Once the object has been saved, the image is sent as a service message to the artificial intelligence node and the callback blocks until the response from the requested service node is received. The stored number serves as a validation for the predicted number in the image from the neural network node.
If the service receives a response from the neural network node, it prints the received digit on the screen, otherwise an error message occurrs that no digit was received.

### Neural network node
The neural network node consists of two parts, the service and the underlying neural network which is responsible for the image recognition.</br>
Before the actual image recognition process, the neural network has to be provided the MNIST(2) datasets. This is needed to train it and evaluate the accuracy of the training run.</br>
The neural network is built up as sequential(10) linear(11) network, where the three hidden layers are connected in a cascading way. The input layer contains 784 neurons. Each neuron stands for one pixel of the image to be recognized. The three hidden layers reduce the number of neurons gradually, up to the output layer which contains 10 neurons for the classification of the predicted number. 
```python
self.input_layer = nn.Sequential(nn.Linear(28 * 28, 512)) 
self.hidden_layer2 = nn.Linear(254, 128)
self.hidden_layer3 = nn.Linear(128, 64)
self.output_layer = nn.Linear(64, 10
```
Each neuron computes a set of input values and weights and an activation function to an output value which is then passed on as input value to next neuron. </br>
Once the network is initialized the next step is to train it. The training function creates an optimizer object with the SGD(12) algorithm and a cross entropy loss function(12). The cross entropy helps to classify the model by outputting the probabiliy values between 0 and 1. SGD(13) stands for stochastic gradient descent. 
The basic functionality of a gradient descent procedure is to find the lowest point of a mathematical function by iterating in steps. To find the lowest point, a random starting point is chosen.</br>
θ = θ − η · ∇θJ(θ) (14)
</br>
Based on the starting point θ, the product of the learning rate η and the result of the cross entropy ∇θJ(θ) is subtracted from the current position θ for each new position. That means the closer the function minimum the smaller the steps become. </br>
Stochastic only means that the starting data point is chosen randomly. Furthermore the entire data set is divided into small batches to minimize the computations and increase the variance.

```python
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
```
Each iteration clears the gradients from the previous to update the parameters correctly. Now the Tensor is passed to the forward function, which flattens the input to a one dimensional Tensor. Then, each neuron in the hidden layers processes the input and weight and the rectified linear activation functions to a new output Tensor. The following loss function computes the gradients and the optimizer updates the weights during the backpropagation.
```python
...
for images, labels in self.training_data:
    optimizer.zero_grad() 
    images, labels = images.to(self.device), labels.to(self.device)
    output = self.model(images)
    loss = criterion(output, labels)
    loss.backward() 
    optimizer.step() 
    ...
```
To evaluate the trained model a verification is perfomed. This gives an overview if the model is robust, under- or overfitted.
</br></br>
When the evaluation is complete the model is saved to the project folder. If the node locates a saved model in the specified folder the next time it is launched, the service server is launched and the node is ready to receive images. The incoming service message contains the image as a ROS sensor message. The callback function is wrapped in a lambda function which allows to take the service object as additional argument.
```python
rospy.Service('image_ack', ImageAck, lambda request : callback (request, ai_service))
```
The prediction function sets the mode to evaluation for the trained and loaded model. 
In order to use the ROS(3) sensor message image properly, it must be converted to PyTorch's Tensor format and normalized to the same values the trained model is. Now the image is passed to the trained model object and the neural network returns the vector of raw predictions that a classification model generates. Every prediction gets processed on the cpu, because the ```numpy```(15) module is not cuda compatible and the tensor vector needs to be converted to a numpy(15) vector to return the largest predicted probability of the digit in the image. The service callback function sends the predicted digit back to the controller node.
</br>
### Graph
An overview of the arrangement of all nodes in the application.</br>
![Network graph](https://github.com/Telespielstube/imagineer/blob/docu/media/network_graph.png)
Figure 1: Graph of all nodes in the robot application.

### Conclusion
The specification of the project was to create a robot application connected to a neurarl network to recognize handwritten digits.</br>
The approach to separate the different tasks makes it easier to maintain each single node and and ensures the ability to extent the application.</br> 
Building the neural network with three hidden layers was based on the consideration that on the one hand there was a rather simple prediction problem, maintain a good performance without overfitting the model and on the other hand to ensure a gradual reduction of neurons in the layers as well. Regarding the rather simple task, the use of the SGD(13) optimizer gives a good insight understanding the basics of optimizers.</br>
</br></br>
|     | Trainig loss | Validation loss | Accuracy |
|-----|--------------|-----------------|----------|
| SGD | 0.0003014007 |     0.0005      |  98.14%  |
</br>Figure 2: Output of the complete training run and validation with the SGD optimizer.</br>
With the MNIST database a robust model can be achieved with little effort. A robust model is defined by the fact that training and validation results are close together (see Figure 1). As seen in the table, the training loss and validation loss are very close. This result was achieved with only 10 epochs a batch size of 32 a standard learning rate of 0.001 and a five-layer fully connected model.
That points to the conclusion that SGD(13) is a very reliable and highly accurate methode for small test applications.

### Sources

1. C++[https://www.cplusplus.com]</br>
2. Python [https://www.python.org]</br>
3. ROS [https://www.ros.org]</br>
4. PyTorch [https://pytorch.org]
5. ROS Messages [http://wiki.ros.org/Messages]</br>
6. ROS Service [http://wiki.ros.org/Services]</br>
7. roslaunch[http://wiki.ros.org/roslaunch]</br>
8. OpenCV [https://opencv.org]
9. MNIST [http://yann.lecun.com/exdb/mnist/]</br>
10. PyTorch Sequential [https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html] </br>
11. PyTorch Linear [https://pytorch.org/docs/stable/generated/torch.nn.Linear.html]</br>
12. PyTorch CrossEntropyLoss [https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html]</br>
13. PyTorch SGD [https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD]</br>
14. An overview of gradient descent optimization algorithms [https://arxiv.org/pdf/1609.04747.pdf]
15. NumPy [https://numpy.org]

### Figures
Figure 1: Graph of all nodes in the robot application.</br>
Figure 2: Output of the complete training run and validation with the SGD optimizer.