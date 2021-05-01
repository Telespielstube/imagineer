#include <ros/ros.h>
#include <iterator>
#include <iostream>
#include <map>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int32.h>
#include "imagineer.srv/ImageAck.h"

void send_image_ack()
{

}

/* adds the subscribed messages as key value pairs to a map.
* @image_message    contains the image received from the subcribed camera/image topic   
* @int_message
* @storage          map<> data structure to save the messages from the topics as key value pairs.
*/
inline void add_to_map(const sensor_msgs::ImageConstPtr& image_message, 
                       const std_msgs::Int32 int_message, 
                       std::map<sensor_msgs::ImageConstPtr, 
                       std_msgs::Int32>& storage)
{
    storage.insert(std::pair<sensor_msgs::ImageConstPtr, std_msgs::Int32>(image_message, int_message))
}

/* Callback function which is called when the node rerceives a new message from subscrribed topics.
* @image_message    contains the image received from the subcribed camera/image topic   
* @int_message
* @storage          map<> data structure to save the messages from the topics as key value pairs.
*/
void callback(const sensor_msgs::ImageConstPtr& image, 
              const std_msgs::Int32 number, 
              std::map<sensor_msgs::ImageConstPtr, std_msgs::Int32>& storage,
              ros::ServiceClient ack_message)
{
    try
    {
        add_to_map(image_message, int_message, storage);
        ROS_INFO("Int and image are saved");
        send_ack_message(image_message, ack_message);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Something went wrong");
    }
}

/* Entry point for the software program.
* @argc    command line passed argument count and that the number of parameters passed
* @argv    command line passed argument values. This contains the images passed from the command line 
*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "controller");
    ros::NodeHandle node;
    std::map<sensor_msgs::ImageConstPtr, std_msgs::Int32> storage;
    ros::ServiceClient ack_message = node.serviceClient<imagineer::ImageAck>("ImageAck");
    message_filters::Subscriber<sensor_msgs::Image> img_subscriber(node, "processor/image", 1);
    message_filters::Subscriber<std_msgs::Int32> int_subscriber(node, "camera/integer", 1); 
    message_filters::TimeSynchronizer<sensor_msgs::ImageConstPtr, std_msgs::Int32> sync(img_subscriber, int_subscriber); 
    sync.registerCallback(boost::bind(callback, _1, _2, storage, ack_message); // boost::bind() allows to pass arguments to a callback. E.g. map<> 
    
    // imagineer:ImageAck ack_service;
    // ack_service.request.img = std::map<sensor_msgs::ImageConstPtr;
    
    ros::spin();
}

