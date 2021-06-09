#include "controller.h"

void Controller::send_image()
{       
    imagineer::ImageAck service;
    int number = 0;
    if (!storage.empty())
    {
        service.request.image = storage.back().get_image(); // image gets passed to the service request image attribute.
        number = storage.back().get_number(); // the corresponding label(number) gets passed to an integer. 
    }
    if (service_client.call(service))
    {    
        if ((int)service.response.result == number) 
        {
            ROS_INFO("Prediction was successful %i", (int)service.response.result);
        }
        else
        {
            ROS_INFO("Prediction was wrong. Received number: %i", (int)service.response.result);
        }
    }
    else
    {
        ROS_ERROR("No number received!");
    }
}

/* Adds received image and corresponding number from camera to the end of a growable list (vector).
* @digit    corresponding image number.
* @image    image object.
*/
inline void Controller::add_to_list(int digit, sensor_msgs::Image& image)
{
    storage.push_back(NumberAndPicture(digit, image));
}

/* Function is called as soon as the node syncronized the received image and number.
* @image
* @digit
*/
void Controller::callback(const sensor_msgs::ImageConstPtr& image, const imagineer::Number& digit)
{
    try 
    {
       // cv::imshow("view", cv_bridge::toCvCopy(image)->image);
       // cv::waitKey(30); 
        int number = digit.digit; // passes the ImageAck digit to an integer
        sensor_msgs::Image save_image = *image; 
        add_to_list(number, save_image);
        send_image();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Error: %s", e.what());
    }
}