#include "number_and_picture.h"
#include <sensor_msgs/Image.h>

int NumberAndPicture::get_number()
{
    return num;
}

void NumberAndPicture::set_number(int number)
{
    num = number;
}

sensor_msgs::ImageConstPtr NumberAndPicture::get_image()
{
    return img;
}

void NumberAndPicture::set_image(sensor_msgs::ImageConstPtr& image)
{
    img = image;
}