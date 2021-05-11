#include "image.h"

int Image::get_name() 
{
    return name;
}

void Image::set_name(int integer)
{
    name = integer;
}

/* Getter function for image content.
* @return     content of Image object.
*/
cv::Mat Image::get_image() 
{
    return image;
}

void Image::set_image(cv::Mat img)
{
    image = img;
}