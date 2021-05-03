#include <ros/ros.h>
#include <iterator>
#include <iostream>
#include <map>
#include <cv_bridge/cv_bridge.h>
#include <boost/bind.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int32.h>
#include "imagineer/ImageAck.h"


class Controller
{
    public:

        ros::NodeHandle node;
        imagineer::ImageAck ack_service;
        std::map<sensor_msgs::ImageConstPtr, std_msgs::Int32> storage;
        message_filters::Subscriber<sensor_msgs::Image> img_subscriber; 
        message_filters::Subscriber<std_msgs::Int32> int_subscriber;
        message_filters::TimeSynchronizer<sensor_msgs::Image, std_msgs::Int32> sync;

        Controller() : sync(img_subscriber, int_subscriber, 1)
        {
            ros::ServiceClient service_client = node.serviceClient<imagineer::ImageAck>("ImageAck");
            img_subscriber.subscribe(node, "processor/image", 1);
            int_subscriber.subscribe(node, "camera/integer", 1);  
            sync.registerCallback(boost::bind(&Controller::callback, this, _1, _2, _3)(storage, ack_service, service_client)); // boost::bind() allows to pass arguments to a callback. E.g. a map<int, string> 
        }

        /* Sends the image as servide message to the neural network node.
        * @image             message to be send to the neural network node.
        * @service_client    Service object.
        * @ack_service       Service message object.
        */
        void send_image(const sensor_msgs::ImageConstPtr& image, 
                    ros::ServiceClient service_client,
                    imagineer::ImageAck ack_service)
        {     
            sensor_msgs::Image ai_image_message;
            ai_image_message = *image; // passes ImageConstPtr to sensor_msg format
            ack_service.request.image = ai_image_message;
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

        /* Callback function which is called when the node receives a new message from subscribed topics.
        * @image_message    contains the image received from the subcribed camera/image topic   
        * @int_message
        * @storage          map<> data structure to save the messages from the topics as key value pairs.
        */
        void callback(const sensor_msgs::ImageConstPtr& image, 
                    const std_msgs::Int32& number, 
                    std::map<sensor_msgs::ImageConstPtr, std_msgs::Int32>& storage,
                    imagineer::ImageAck ack_service,
                    ros::ServiceClient service_client)
        {
            try
            {
                add_to_map(image, number, storage);
                ROS_INFO("Int and image are saved");
                send_image(image, service_client, ack_service);
            }
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("Error: %s", e.what());
            }
        }
};

/* Entry point for the software program.
* @argc    command line passed argument count and that the number of parameters passed
* @argv    command line passed argument values. This contains the images passed from the command line 
*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "controller");
    Controller controller;

    ros::spin();
}

