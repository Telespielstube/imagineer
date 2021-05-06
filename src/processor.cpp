#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



class Procesor
{
    public:
        Processor()
        {
            ros::NodeHandle node;
            image_transport::ImageTransport transport(node);
            subscriber = transport.subscribe("camera/image", 1, callback);
            publisher = transport.advertise("processor/image", 1);
        }

        /* Does all the image processing like resizeing and grayscaling the original image.
        * @message    contains a reference to the original image.
        */
        cv::Mat process_image(cv::Mat& message)
        {
            cv::resize(message, message, cv::Size(), 0.5, 0.5, CV_INTER_AREA);
            cv::cvtColor(message, message, cv::COLOR_BGR2GRAY);
            cv::Mat processed_image = message;
            return processed_image;
        }

        /* If a new message arrives on the subscribed topic this function gets called.
        * @message    contains the original image.
        */
        void callback(const sensor_msgs::ImageConstPtr& message)
        {
            cv::Mat img_message = cv_bridge::toCvCopy(message)->image; // Converts the cv_bridge back to a ros image.
            try
            {
                cv::Mat processed_image = process_image(img_message);
                publisher.publish(img_message);
                ROS_INFO("Image is published.");
            }
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("Could not convert from '%s' to 'bgr8'", message->encoding.c_str());
            }
        }

    private:
        image_transport::Subscriber subscriber;
        image_transport::Publisher publisher;

};
/* Callback function which is called when the node rerceives a new message from subscrribed topics.
* @image_message    contains the image received from the subcribed camera/image topic   
* @int_message
* @storage          map<> data structure to save the messages from the topics as key value pairs.
*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "processor");
    ROS_INFO("Processor node is running");
    
    ros::spin();
}
