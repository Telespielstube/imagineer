#include <ros/ros.h>
#include "controller.h"

/* Entry point for the software program.
* @argc    command line passed argument count and that the number of parameters passed
* @argv    command line passed argument values. This contains the images passed from the command line 
*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "controller");
    ROS_INFO("Controller node is running");

    Controller controller;

    ros::spin();
    cv::destroyWindow("view");
}
