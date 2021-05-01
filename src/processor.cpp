#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
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
void callback(const sensor_msgs::ImageConstPtr& message)
{
    cv::Mat original_msg = cv_bridge::toCvCopy(message)->image; // Converts the cv_bridge back to a ros image.
    try
    {
        cv::Mat processed_image = process_image(original_msg);

        cv::imshow("view", processed_image);
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'", message->encoding.c_str());
    }
}

/* Callback function which is called when the node rerceives a new message from subscrribed topics.
* @image_message    contains the image received from the subcribed camera/image topic   
* @int_message
* @storage          map<> data structure to save the messages from the topics as key value pairs.
*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "processor");
    ROS_INFO("Processor node is running");
    ros::NodeHandle node;
    cv::namedWindow("view");

    image_transport::ImageTransport transport(node);
    image_transport::Publisher publisher = transport.advertise("processor /image", 1);
    image_transport::Subscriber subscriber = transport.subscribe("camera/image", 1, callback);

    ros::spin();
    cv::destroyWindow("view");
}
