#include "controller.h"

void Controller::send_image()
{     
    imagineer::ImageAck ack_service;
    ack_service.request.image = storage.back().get_image();
    if (service_client.call(ack_service))
    {
        ROS_INFO("Received number: %i", (int)ack_service.response.result);
        // compare_result();
    }
    else
    {
        ROS_ERROR("No number received!");
    }
}

void Controller::add_to_list(int digit, sensor_msgs::Image& image)
{
    storage.push_back(NumberAndPicture(digit, image));
}

void Controller::callback(const sensor_msgs::ImageConstPtr& image, const imagineer::Number& digit)
{
    try
    {
        cv::imshow("view", cv_bridge::toCvCopy(image)->image);
        cv::waitKey(30); 
        int number = digit.digit;
        sensor_msgs::Image save_image = *image;
        add_to_list(number, save_image);
        send_image();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Error: %s", e.what());
    }
}



