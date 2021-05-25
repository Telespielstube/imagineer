#include "number_and_picture.h"

int NumberAndPicture::get_number()
{
    return num.digit;
}

void NumberAndPicture::set_number(int number)
{
    num.digit = number;
}

sensor_msgs::Image NumberAndPicture::get_image()
{
    return img;
}

void NumberAndPicture::set_image(sensor_msgs::Image& image)
{
    img = image;
}