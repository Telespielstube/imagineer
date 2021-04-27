#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int32.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Does all the image processing like resizeing and grayscaling the original image.
* @message    contains the original image.
*/
cv::Mat process_image(cv::Mat& message)
{
    cv::Mat processed_image;
    cv::resize(message, message, cv::Size(), 0.5, 0.5, cv::CV_INTER_AREA);
    cv::cvtColor(message, message, cv::COLOR_BGR2GRAY);
    //cv::threshold(message, message, 80, 255, cv::THRESH_BINARY);
    processed_image = message;
    return processed_image
}

/* If a new message arrives on the subscribed topic this function gets called.
* @message    contains the original image.
*/
void img_callback(const sensor_msgs::ImageConstPtr& message)
{
    cv::Mat original_msg = cv_bridge::toCvCopy(message)->image; // Converts the cv_bridge back to a ros image.
    try
    {
        //cv::resize(original_msg, original_msg, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        //cv::cvtColor(original_msg, original_msg, cv::COLOR_BGR2GRAY);
        //cv::threshold(original_msg, original_msg, 150, 255, cv::THRESH_BINARY);
        cv::Mat processed_image = process_image(original_msg);
        cv::imshow("view", processed_image);
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'", message->encoding.c_str());
    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "processor");
    ros::NodeHandle node;
    cv::namedWindow("view");

    image_transport::ImageTransport transport(node);
    image_transport::Subscriber sub = transport.subscribe("camera/image", 1, img_callback);
    ros::spin();
    cv::destroyWindow("view");
}
