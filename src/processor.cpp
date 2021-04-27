#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int32.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// /* Resizes the original image.
// * @message    contains the original image.
// */
// inline cv::Mat resize_image(cv::Mat& message)
// {
//     return cv::resize(message, message, cv::Size(), 0.5, 0.5, cv::CV_INTER_AREA);
// }

// // /* Converts resized immage to grayscale image. 
// // * @message    contains the original image.
// // */
// inline cv::Mat convert_to_grayscale(cv::Mat& message)
// {
//     return cv::cvtColor(message, message, cv::COLOR_BGR2GRAY);
// }

// // /* Adds threshold to the resized and grayscale image. 
// // * @message    contains the original image.
// // */
// inline cv::Mat add_threshold(cv::Mat &message)
// {
//     return cv::threshold(message, message, 80, 255, cv::THRESH_BINARY);
// }

/* If a new message arrives on the subscribed topic this function gets called.
* @message    contains the original image.
*/
void img_callback(const sensor_msgs::ImageConstPtr& message)
{
    cv::Mat original_msg = cv_bridge::toCvCopy(message)->image; // Converts the cv_bridge back to a ros image.
    try
    {
        // cv::Mat resized_img = cv::resize_image(original_msg);
        // cv::Mat grayscal_image = cv::cvtColor(original_msg, original_msg, cv::COLOR_BGR2GRAY);
        // cv::Mat threshold_image = cv::threshold(original_msg, original_msg, 150, 255, cv::THRESH_BINARY);
        cv::resize(original_msg, original_msg, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        cv::cvtColor(original_msg, original_msg, cv::COLOR_BGR2GRAY);
   //     cv::threshold(original_msg, original_msg, 150, 255, cv::THRESH_BINARY);
        cv::imshow("view", original_msg);
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'", message->encoding.c_str());
    }
}

void int_callback(const std_msgs::Int32 message)
{
    try
    {
        ROS_INFO("%d", message.data);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Something went wrong");
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "processor");
    ros::NodeHandle node;
    cv::namedWindow("view");

    image_transport::ImageTransport transport(node);
    image_transport::Subscriber sub = transport.subscribe("camera/image", 1, img_callback);
    ros::Subscriber subscriber = node.subscribe("image/integer", 1, int_callback);
    ros::spin();
    cv::destroyWindow("view");
}
