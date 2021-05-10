#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class Processor
{
    public:
        Processor() : transport(node)
        {
            subscriber = transport.subscribe("camera/image", 1, &Processor::callback, this);
            publisher = transport.advertise("processor/image", 1);
        }

        /* Does all the image processing like resizeing and grayscaling the original image.
        * @message    contains a reference to the original image.
        */
        inline cv::Mat process_image(cv::Mat& message)
        {
            cv::resize(message, message, cv::Size(), 0.5, 0.5, CV_INTER_AREA);
            cv::Mat processed_image = message;
            return processed_image;
        }

        /* Callback function which is called when the node rerceives a new message from subscrribed topics.
        * @message    contains an image reference to the received image;   
        */
        void callback(const sensor_msgs::ImageConstPtr& message)
        {
            try
            {
                cv::Mat resized_image = process_image(cv_bridge::toCvCopy(message)->image); // Converts the cv_bridge back to a ros image and processes it.
                publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", resized_image).toImageMsg()); 
                ROS_INFO("Image is published from processor node.");
            }
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("Could not convert from '%s' to 'bgr8'", message->encoding.c_str());
            }
        }

    private:
        ros::NodeHandle node;
        image_transport::ImageTransport transport;
        image_transport::Subscriber subscriber;
        image_transport::Publisher publisher;

};

/* Entry point for the software program.
* @argc    command line passed argument count and that the number of parameters passed
* @argv    command line passed argument values. This contains the images passed from the command line 
*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "processor");
    ROS_INFO("Processor node is running");
    
    Processor processor;

    ros::spin();
}
