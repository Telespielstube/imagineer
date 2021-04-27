#include <ros/ros.h>
#include "imagineer/ImageAck.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "controller");
    ros::NodeHandle node;

    ros::ServiceServer service = node.advertiseService("")
}