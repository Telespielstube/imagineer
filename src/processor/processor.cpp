#include "processor.h"
#include <opencv2/highgui/highgui.hpp>

cv::Mat Processor::process_image(cv::Mat& message)
{
    cv::Mat resized_message;
    cv::Mat gray;
    cv::Mat invert;
    cv::resize(message, message, cv::Size(28, 28));
    cv::cvtColor(message, message, cv::COLOR_BGR2GRAY);
    cv::threshold(message, message, 100, 255, cv::THRESH_BINARY);
    cv::bitwise_not(message, message);
    return message;
}

void Processor::callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::namedWindow("view", cv::WINDOW_AUTOSIZE);

        cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", resized_image).toImageMsg()); 
        ROS_INFO("Image is published.");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert received image %s.", message->encoding.c_str());
       // ROS_ERROR("Could not convert received image %s. Detailed error message: %c", e.what());
    }
}