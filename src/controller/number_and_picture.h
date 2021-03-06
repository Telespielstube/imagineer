#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "imagineer/Number.h"

class NumberAndPicture
{
    public:
        // two contructors, first one is the default constructor, second one expects 2 arguments.
        NumberAndPicture() {} 
        NumberAndPicture(int digit, const sensor_msgs::Image& image)
        {
            num = digit;
            img = image;
        }
        
        /* operator overloading function which takes argument &other and copies it to a memeber variable.
        * @other        reference to a parameter to be copied to a member variable .
        * @return       object reference.
        */ 
        NumberAndPicture& operator= (const NumberAndPicture &other)
        {
            num = other.num;
            img = other.img;
            return *this;
        }

        /* Returns the number associated to the image.
        * @return    number
        */
        int get_number();
        
        /* Sets the number to the associated image.
        */
        void set_number(int number);

        /* Returns the image object.
        * @return    image object.
        */
        sensor_msgs::Image get_image();

        /* Sets the image object.
        */
        void set_image(sensor_msgs::Image& image);
       
    private:
        int num;
        sensor_msgs::Image img;
};