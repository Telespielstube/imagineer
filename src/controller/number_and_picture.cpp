#include "number_and_picture.h"

int NumberAndPicture::get_number()
{
    return num;
}

void NumberAndPicture::set_number(int number)
{
    num = number;
}

sensor_msgs::Image NumberAndPicture::get_image()
{
    return img;
}

void NumberAndPicture::set_image(sensor_msgs::Image& image)
{
    img = image;
}