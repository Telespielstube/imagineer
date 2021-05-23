#include "processor.h"

// cv::Mat Processor::process_image(cv::Mat& message)
// {
//     return cv::resize(message, message, cv::Size(), 0.5, 0.5, CV_INTER_AREA);
//     cv::Mat processed_image = message;
//     return processed_image;
// }

void Processor::callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        ROS_INFO("Message %s received", message->encoding.c_str());
        //cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
        cv::Mat cv_image = cv_bridge::toCvCopy(message)->image;
        cv::resize(cv_image, cv_image, cv::Size(), 0.5, 0.5, CV_INTER_AREA);
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", cv_image).toImageMsg()); 
        ROS_INFO("Image is published from processor node.");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert received image %s", message->encoding.c_str());
    }
}