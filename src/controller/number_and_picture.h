#include <sensor_msgs/Image.h>
#include "imagineer/Number.h"
#include "imagineer/ImageAck.h"

class NumberAndPicture
{
    public:
        // two contructors, first one is the default constructor, second one expects 2 arguments.
        NumberAndPicture() {} 
        NumberAndPicture(const imagineer::Number digit, sensor_msgs::Image image)
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

    private:
        imagineer::Number num;
        sensor_msgs::Image img;
};