#include "processor.h"
#include <opencv2/highgui/highgui.hpp>

cv::Mat Processor::process_image(cv::Mat& message)
{
    cv::Mat resized_message;
    cv::resize(message, resized_message, cv::Size(28, 28));
    return resized_message;
}

cv::Mat Processor::color_to_grey(cv::Mat& color_image)
{
    cv::Mat greyscale;
    cv::cvtColor(color_image, greyscale, CV_BGR2GRAY);
    return greyscale;
}
void Processor::callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::namedWindow("view", cv::WINDOW_AUTOSIZE);

        cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
        cv::Mat greyscale =  color_to_grey(resized_image);
        // cv::imshow("view", greyscale);
        // cv::waitKey(30); 
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", resized_image).toImageMsg()); 
        ROS_INFO("Image is published.");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert received image %s.", message->encoding.c_str());
       // ROS_ERROR("Could not convert received image %s. Detailed error message: %c", e.what());
    }
}