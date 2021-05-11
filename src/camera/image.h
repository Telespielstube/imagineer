#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

class Image
{
    public: 
        Image() {}
        Image(int filename, cv::Mat content) 
        {
            name = filename;
            image = content;
        }
        
        /* operator overloading function which takes argument &other and copies it to a memeber variable.
        * @other        reference to a parameter to be copied to a member variable .
        * @return       object reference.
        */ 
        Image& operator= (const Image &other)
        {
            name = other.name;
            image = other.image;
            return *this;
        }

        /* Getter function for image filename.
        * @return     filename of Image object.
        */
        int get_name();

        /* Setter function for image filename.
        * @integer    sets the filename for the corresponding image
        */
        void set_name(int integer);

        /* Getter function for image content.
        * @return     content of Image object.
        */
        cv::Mat get_image();

        /* Setter function for image content.
        * @img    sets the content of the image.
        */
        void set_image(cv::Mat img);

    private:
        int name;
        cv::Mat image;
};