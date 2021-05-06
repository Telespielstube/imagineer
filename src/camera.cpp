#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <std_msgs/Int32.h>
#include "imagineer/Number.h"

// class Camera
// {
//     public:
    
// };
/* Callback function which is called when the node rerceives a new message from subscrribed topics.
* @image_message    contains the image received from the subcribed camera/image topic   
* @int_message
* @storage          map<> data structure to save the messages from the topics as key value pairs.
*/
int main(int argc, char** argv)
{
    ros::init(argc, argv, "camera");
    ROS_INFO("Camera node is running");
    ros::NodeHandle node;

    image_transport::ImageTransport transport(node);
    image_transport::Publisher img_publisher = transport.advertise("camera/image", 1);
    ros::Publisher int_publisher = node.advertise<imagineer::Number>("camera/integer", 1);
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    sensor_msgs::ImagePtr img_message = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    std_msgs::Header header;
    imagineer::Number int_message; 
    int_message.digit = 2;

    ros::Rate loop(50);
    // as long as the node is running and at least one node
    // subscribes to the two topics the camera node sends both messages.
    while (node.ok()) 
    {
        if (img_publisher.getNumSubscriber() > 0 && int_publisher.getNumSubscriber() > 0)
        {
            img_publisher.publish(img_message);
            int_publisher.publish(int_message);
        }
        else{ 
            continue;
        }
        ros::spinOnce();
        loop.sleep();
    }
}