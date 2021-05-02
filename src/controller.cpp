#include <ros/ros.h>
#include <iterator>
#include <iostream>
#include <map>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int32.h>
#include "imagineer/ImageAck.h"

void send_image_ack(const sensor_msgs::ImageConstPtr& image, 
                    ros::ServiceClient service_client,
                    imagineer::ImageAck ack_service)
{
   // cv::Mat ros_image = cv_bridge::toCvCopy(image)->image;
    //ack_service.request.image = image;
    if (service_client.call(ack_service))
    {
        ROS_INFO("Received number: %d", ack_service.response.number);
    }
    else
    {
        ROS_ERROR("Something went wrong no number received!");
    }
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
    storage.insert(std::pair<sensor_msgs::ImageConstPtr, std_msgs::Int32>(image_message, int_message));
}

/* Callback function which is called when the node rerceives a new message from subscrribed topics.
* @image_message    contains the image received from the subcribed camera/image topic   
* @int_message
* @storage          map<> data structure to save the messages from the topics as key value pairs.
*/
void callback(const sensor_msgs::ImageConstPtr& image, 
            const std_msgs::Int32 number, 
            std::map<sensor_msgs::ImageConstPtr, std_msgs::Int32>& storage,
            imagineer::ImageAck ack_service,
            ros::ServiceClient service_client)
{
    try
    {
        add_to_map(image, number, storage);
        ROS_INFO("Int and image are saved");
        send_image_ack(image, service_client, ack_service);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Error: %s", e.what());
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

    ros::ServiceClient service_client = node.serviceClient<imagineer::ImageAck>("ImageAck");
    imagineer::ImageAck ack_service;

    message_filters::Subscriber<sensor_msgs::Image> img_subscriber(node, "processor/image", 1);
    message_filters::Subscriber<std_msgs::Int32> int_subscriber(node, "camera/integer", 1); 
    message_filters::TimeSynchronizer<sensor_msgs::Image, std_msgs::Int32> sync(img_subscriber, int_subscriber, 2); 
    sync.registerCallback(boost::bind(callback, _1, _2, _3, storage, ack_service, service_client)); // boost::bind() allows to pass arguments to a callback. E.g. map<> 
    
    ros::spin();
}

