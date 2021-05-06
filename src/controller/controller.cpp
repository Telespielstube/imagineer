#include "controller.h"

/* Sends the image as servide message to the neural network node.
* @image             message to be send to the neural network node.
* @ack_service       Service message object.
*/
void Contrroller::send_image(const sensor_msgs::ImageConstPtr& image, imagineer::ImageAck ack_service)
{     
    sensor_msgs::Image ai_message = *image; // passes ImageConstPtr to sensor_msg format
    ack_service.request.image = ai_message;
    if (service_client.call(ack_service))
    {
        ROS_INFO("Received number: %d", ack_service.response.number);
    }
    else
    {
        ROS_ERROR("Something went wrong no number received!");
    }
}

/* adds the subscribed messages as key value pairs to a map.
* @image_message    contains the image received from the subcribed camera/image topic   
* @int_message      contains the number received from the subcribed camera/integer topic.   
*/
void Controller::add_to_list(const imagineer::Number digit, const sensor_msgs::ImageConstPtr image)
{
    sensor_msgs::Image saved_image = *image;
    storage.push_back(NumberAndPicture(digit, saved_image));
}

/* Callback function which is called when the node receives a new message from subscribed topics.
* @image    contains the image received from the subcribed camera/image topic.   
* @digit    contains the number received from the subcribed camera/integer topic.   
*/
void Controller::callback(const sensor_msgs::ImageConstPtr& image, const imagineer::Number& digit)
{
    try
    {
        cv::Mat original_msg = cv_bridge::toCvCopy(image)->image;
        cv::imshow("view", original);
        cv::waitKey(30);
        imagineer::ImageAck ack_service;
        add_to_list(digit, image);
        ROS_INFO("Int and image are saved");
        send_image(image, ack_service);
        ROS_INFO("Image sent");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Error: %s", e.what());
    }
}