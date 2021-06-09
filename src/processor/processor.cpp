#include "processor.h"
#include <opencv2/highgui/highgui.hpp>

cv::Mat Processor::process_image(cv::Mat& message)
{
    float scale_down = 0.5;
    cv::Mat resized_message;
    cv::resize(message, resized_message, cv::Size(message.cols * scale_down, message.rows * scale_down), 0, 0);
    return resized_message;
}

void Processor::callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::namedWindow("view", cv::WINDOW_AUTOSIZE);

        cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
        cv::imshow("view", resized_image);
        cv::waitKey(30); 
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", resized_image).toImageMsg()); 
        ROS_INFO("Image is published.");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert received image %s. Error: %s", message->encoding.c_str());
       // ROS_ERROR("Could not convert received image %s. Detailed error message: %c", e.what());
    }
}