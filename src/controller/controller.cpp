#include "controller.h"
#include <iostream>

void Controller::send_image()
{       
    imagineer::ImageAck service
    int number = 0;
    if (!storage.empty())
    {
        service.request.image = storage.back().get_image();
        number = storage.back().get_number();
    }
    if (service_client.call(service))
    {
        
        ROS_INFO("Received number: %i", (int)service.response.result);
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
    for (auto i : storage)
    {
        std::cout << storarge.at(i) << std::endl;
    }
}

void Controller::callback(const sensor_msgs::ImageConstPtr& image, const imagineer::Number& digit)
{
    try 
    {
        cv::imshow("view", cv_bridge::toCvCopy(image)->image);
        cv::waitKey(30); 
        int number = digit.digit; // passes the ImageAck filed digit 
        sensor_msgs::Image save_image = *image;
        add_to_list(number, save_image);
        send_image();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Error: %s", e.what());
    }
}



