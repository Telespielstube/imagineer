#include <iterator>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <boost/bind.hpp>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <opencv2/highgui/highgui.hpp>
#include "imagineer/Number.h"
#include "imagineer/ImageAck.h"
#include "number_and_picture.h"

class Controller
{
    public:    
        /* Contructor
        */ 
        Controller() 
        {
            img_subscriber.subscribe(node, "processor/image", 1);
            int_subscriber.subscribe(node, "camera/integer", 1); 
            cv::namedWindow("view");
            sync.reset(new message_filters::TimeSynchronizer<sensor_msgs::Image, imagineer::Number>(img_subscriber, int_subscriber, 10));
            service_client = node.serviceClient<imagineer::ImageAck>("ImageAck", true);
            sync->registerCallback(&Controller::callback, this);   
        }

        /* Sends the image as servide message to the neural network node.
        * @image             message to be send to the neural network node.
        * @ack_service       Service message object.
        */
        void send_image(const sensor_msgs::ImageConstPtr& image);

        /* adds the subscribed messages as key value pairs to a map.
        * @image_message    contains the image received from the subcribed camera/image topic   
        * @int_message      contains the number received from the subcribed camera/integer topic.   
        */
        void add_to_list(const imagineer::Number digit, const sensor_msgs::ImageConstPtr image);

        /* Callback function which is called when the node receives a new message from subscribed topics.
        * @image    contains the image received from the subcribed camera/image topic.   
        * @digit    contains the number received from the subcribed camera/integer topic.   
        */
        void callback(const sensor_msgs::ImageConstPtr& image, const imagineer::Number& digit);
        
        private:
            ros::NodeHandle node;
            ros::ServiceClient service_client;
            std::vector<NumberAndPicture> storage;
            message_filters::Subscriber<sensor_msgs::Image> img_subscriber; 
            message_filters::Subscriber<imagineer::Number> int_subscriber;
            boost::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::Image, imagineer::Number>> sync;
};