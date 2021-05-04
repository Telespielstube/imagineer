#include <ros/ros.h>
#include <iterator>
#include <iostream>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <boost/bind.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include "imagineer/Number.h"
#include "imagineer/ImageAck.h"


class NumberAndPicture
{
    public:
        NumberAndPicture() {} // default constructor
        NumberAndPicture(const imagineer::Number digit, sensor_msgs::Image image)
        {
            num = digit;
            img = image;
        }
        
        NumberAndPicture& operator= (const NumberAndPicture &other)
        {
            num = other.num;
            img = other.img;
            return *this;
        }

    private:
        imagineer::Number num;
        sensor_msgs::Image img;

};

class Controller
{
    public:
        ros::NodeHandle node;
        ros::ServiceClient service_client;
        std::vector<NumberAndPicture> storage;
        message_filters::Subscriber<sensor_msgs::Image> img_subscriber; 
        message_filters::Subscriber<imagineer::Number> int_subscriber;
       // message_filters::TimeSynchronizer<sensor_msgs::Image, imagineer::Number> sync;

        Controller() //: sync(img_subscriber, int_subscriber, 1)
        {
            service_client = node.serviceClient<imagineer::ImageAck>("ImageAck");
            img_subscriber.subscribe(node, "processor/image", 1);
            int_subscriber.subscribe(node, "camera/integer", 1); 
            message_filters::TimeSynchronizer<sensor_msgs::Image, imagineer::Number> sync(img_subscriber, int_subscriber, 1);
            sync.registerCallback(boost::bind(&Controller::callback)); // boost::bind() allows to pass arguments to a callback. E.g. a map<int, string> 
        }

        /* Sends the image as servide message to the neural network node.
        * @image             message to be send to the neural network node.
        * @service_client    Service object.
        * @ack_service       Service message object.
        */
        // void send_image(const sensor_msgs::ImageConstPtr& image, imagineer::ImageAck ack_service)
        // {     
        //     sensor_msgs::Image ai_message = *image; // passes ImageConstPtr to sensor_msg format
        //     ack_service.request.image = ai_message;
        //     if (service_client.call(ack_service))
        //     {
        //         ROS_INFO("Received number: %d", ack_service.response.number);
        //     }
        //     else
        //     {
        //         ROS_ERROR("Something went wrong no number received!");
        //     }
        // }

        /* adds the subscribed messages as key value pairs to a map.
        * @image_message    contains the image received from the subcribed camera/image topic   
        * @int_message
        * @map          map<> data structure to save the messages from the topics as key value pairs.
        */
        void add_to_list(const imagineer::Number digit, const sensor_msgs::ImageConstPtr image)
        {
            sensor_msgs::Image saved_image = *image;
            storage.push_back(NumberAndPicture(digit, saved_image));
        }

        /* Callback function which is called when the node receives a new message from subscribed topics.
        * @image_message    contains the image received from the subcribed camera/image topic   
        * @int_message
        * @map          map<> data structure to save the messages from the topics as key value pairs.
        */
        void callback(const sensor_msgs::ImageConstPtr& image, const imagineer::Number& digit)
        {
            try
            {
                add_to_list(digit, image);
                ROS_INFO("Int and image are saved");
                //send_image(image, ack_service);
                


                imagineer::ImageAck ack_service; 
                sensor_msgs::Image ai_message = *image; // converts ImageConstPtr to sensor_msg format
                ack_service.request.image = ai_message;
                if (service_client.call(ack_service))
                {
                    ROS_INFO("Received number: %d", ack_service.response.number);
                }
                else
                {
                    ROS_ERROR("Something went wrong no number received!");
                }





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

