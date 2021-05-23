#include "controller.h"

void Controller::send_image(const sensor_msgs::ImageConstPtr& image, imagineer::ImageAck ack_service)
{     
    sensor_msgs::Image ai_message = *image; // passes ImageConstPtr to sensor_msg format
    ack_service.request.image = ai_message;
    if (service_client.call(ack_service))
    {
        ROS_INFO("Received number: %i", (int)ack_service.response.number);
    }
    else
    {
        ROS_ERROR("Something went wrong no number received!");
    }
}

void Controller::add_to_list(const imagineer::Number digit, const sensor_msgs::ImageConstPtr image)
{
    sensor_msgs::Image saved_image = *image;
    storage.push_back(NumberAndPicture(digit, saved_image));
}

void Controller::callback(const sensor_msgs::ImageConstPtr& image, const imagineer::Number& digit)
{
    try
    {
        cv::namedWindow("view", cv::WINDOW_AUTOSIZE);
        cv::imshow("view", cv_bridge::toCvCopy(image)->image);
        cv::waitKey(30);
        imagineer::ImageAck ack_service;
        add_to_list(digit, image);
        send_image(image, ack_service);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Error: %s", e.what());
    }
}



