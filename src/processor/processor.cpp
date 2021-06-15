#include "processor.h"
#include <opencv2/highgui/highgui.hpp>

cv::Mat Processor::process_image(cv::Mat& message)
{
    cv::Mat resized_message;
    cv::Mat grayscale_image;//declaring a matrix to store converted image//
    cv::Mat binary_image;//declaring a matrix to store the binary image
    cv::Mat inverted_binary_image;
    cv::resize(message, resized_message, cv::Size(28, 28));
   // cv::cvtColor(resized_message, grayscale_image, cv::COLOR_BGR2GRAY);//Converting BGR to Grayscale image and storing it into 'converted' matrix//
    //cv::threshold(resized_message, binary_image, 150, 255, cv::THRESH_BINARY);//converting grayscale image stored in 'converted' matrix into binary image//
    //cv::bitwise_not(binary_image, inverted_binary_image);
    
    return resized_message;
}

void Processor::callback(const sensor_msgs::ImageConstPtr& message)
{
    try
    {
        cv::namedWindow("view", cv::WINDOW_AUTOSIZE);

        cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
        publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", resized_image).toImageMsg()); 
        ROS_INFO("Image is published.");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert received image %s.", message->encoding.c_str());
       // ROS_ERROR("Could not convert received image %s. Detailed error message: %c", e.what());
    }
}