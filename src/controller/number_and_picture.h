#include <ros/ros.h>
#include <sensor_msgs/Image.h>

class NumberAndPicture
{
    public:
        // two contructors, first one is the default constructor, second one expects 2 arguments.
        NumberAndPicture() {} 
        NumberAndPicture(const imagineer::Number digit, const sensor_msgs::ImageConstPtr& image)
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
        sensor_msgs::ImageConstPtr get_image();

        /* Sets the image object.
        */
        void set_image(sensor_msgs::ImageConstPtr& image);
       
    private:
        imagineer::Number num;
        sensor_msgs::Image img;
};