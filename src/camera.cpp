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
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    sensor_msgs::ImagePtr img_message = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
    imagineer::Number int_message;
    int_message.header.stamp = ros::Time::now();  
    int_message.digit = 2;

    ros::Rate loop(20);
    // as long as the node is running send the image and integer messages.
    while (node.ok()) 
    {
    
        img_publisher.publish(img_message);
        int_publisher.publish(int_message);
        ros::spinOnce();
        loop_rate.sleep();
    }
}