#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class Processor
{
    public:
        /* Contructor with initializer list for initializing the image transport
        * object before the rest of the variables gets initialized.
        */
        Processor() : transport(node)
        {
            subscriber = transport.subscribe("camera/image", 1, &Processor::callback, this);
            publisher = transport.advertise("processor/image", 1);
        }

        /* Does all the image processing like resizeing and grayscaling the original image.
        * @message    contains a reference to the original image.
        */
        cv::Mat process_image(cv::Mat& message);

        /* Callback function which is called when the node receives a new message from subscrribed topics.
        * @message    contains an image reference to the received image;   
        */
        void callback(const sensor_msgs::ImageConstPtr& message);

    private:
        ros::NodeHandle node;
        image_transport::ImageTransport transport;
        image_transport::Subscriber subscriber;
        image_transport::Publisher publisher;
};