#include <ros/ros.h>
#include "imagineer/ImageAck.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "controller");
    ros::NodeHandle node;

    ros::Subscriber subscriber = node.subscribe("image/integer", 1, int_callback);
    ros::ServiceServer service = node.advertiseService("")
}

void int_callback(const std_msgs::Int32 message)
{
    try
    {
        ROS_INFO("%d", message.data);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Something went wrong");
    }
}