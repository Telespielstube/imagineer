#include "processor.h"

cv::Mat Processor::process_image(cv::Mat& message)
{
    float scale_down = 0.5;
    cv::Mat resized_message;
    cv::resize(message, resized_message, cv::Size(), scale_down, scale_down);
    return resized_message;
}

void Processor::callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", resized_image).toImageMsg()); 
        ROS_INFO("Image is published.");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert received image %s", message->encoding.c_str());
    }
}