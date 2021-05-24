#include "controller.h"

void Controller::send_image()
{     
    imagineer::ImageAck ack_service;
    //sensor_msgs::Image ai_image = *image; // passes ImageConstPtr to sensor_msg format
    ack_service.request.image = storage.back().get_number();
    if (service_client.call(ack_service))
    {
        ROS_INFO("Received number: %i", (int)ack_service.response.result);

    }
    else
    {
        ROS_ERROR("No number received!");
    }
}

void Controller::add_to_list(const imagineer::Number digit, const sensor_msgs::ImageConstPtr image)
{
    sensor_msgs::Image save_image = *image;
    storage.push_back(NumberAndPicture(digit, save_image));
}

void Controller::callback(const sensor_msgs::ImageConstPtr& image, const imagineer::Number& digit)
{
    try
    {
        cv::imshow("view", cv_bridge::toCvCopy(image)->image);
        cv::waitKey(30); 
        add_to_list(digit, image);
        send_image();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Error: %s", e.what());
    }
}



