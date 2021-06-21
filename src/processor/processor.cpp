#include "processor.h"
#include <opencv2/highgui/highgui.hpp>

/* Function for image processing like resizing, grayscaling, inverting and adding a threshold 
*  @message     image object in Opencv format
*
*  @return      processed image in Opencv format.
*/
cv::Mat Processor::process_image(cv::Mat& message)
{
    cv::Mat resized_message;
    cv::Mat grayscale_image;
    cv::Mat threshold_image;
    cv::Mat inverted_threshold_image;
    cv::resize(message, resized_message, cv::Size(28, 28));
    cv::cvtColor(resized_message, grayscale_image, cv::COLOR_BGR2GRAY);
    cv::threshold(grayscale_image, threshold_image, 110, 255, cv::THRESH_BINARY);
    cv::bitwise_not(threshold_image, inverted_threshold_image); 
    return inverted_threshold_image;
}

/* The callback function is called whenever a new message is received.
*  @message    image object in RSO sensor message format.
*/
void Processor::callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::namedWindow("view", cv::WINDOW_AUTOSIZE);
        cv::Mat processed_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", processed_image).toImageMsg()); 
        ROS_INFO("Image is published.");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert received image %s.", message->encoding.c_str());
    }
}