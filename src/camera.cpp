#include <ros/ros.h>
#include <vector>
#include <experimental/filesystem>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <std_msgs/Int32.h>
#include "imagineer/Number.h"

class Image
{
    public: 
        Image() {}
        Image(const int filename, cv::Mat content) 
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
        int get_name() 
        {
            return name;
        }

        void set_name(int integer)
        {
            name = integer;
        }

        cv::Mat get_image() 
        {
            return image;
        }

        void set_image(cv::Mat img)
        {
            image = img;
        }

    private:
        const int name;
        cv::Mat image;
};
 
/* Reads all available files from the directory.
* @path    Path to directory as an argument from the command line.
* @return  list of all files.
*/
std::vector<std::string> get_files(std::string path)
{
    std::vector<std::string> files;
    std::experimental::filesystem::directory_iterator path_iterator(path);
    for (const auto& entry : path_iterator)
    {
        files.push_back(entry.path().string());
    }
    return files;
}

/* Function to pick a file from the files vector. This file gets published.
*  @files     list of all files in the given directory.
*  @return    the randomly choosen file from the vector.
*/
std::string pick_file(std::vector<std::string> files)
{
    int random_file = 5 + (std::rand() % (9 - 1 + 2));
    std::string pick = "";
    for (const auto& entry : files)
    {
        pick = entry.at(random_file);
    }
    return pick;
}

/* Reads the content of the given file and saves content and filename as an unordered map.
* @image_files    a list of all files from the given command line path.
* @return         an Image object that hold the filename and the image as attributes.
*/
Image read_image(std::string image_file)
{
    Image message;
    const int filename = std::stoi(image_file.substr(16, 17));
    cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
    message.set_name(filename);
    message.set_image(image);
    return message;
}

/* Publishes the key, value pair as std_msgs and imagineer::Number messages to all subscribers.
* @node            Node object.
* @img_publisher   image transport publisher object.
* @int_message     integer publisher object.
# @Image           Image object with filename and image as attributes.
*/
void publish_message(ros::NodeHandle node, image_transport::Publisher img_publisher, ros::Publisher int_publisher, 
                    Image message_to_publish)
{
        imagineer::Number message;
        message.digit = message_to_publish.get_name();
        int_publisher.publish(message.digit);
        img_publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", message_to_publish.get_image()).toImageMsg());       
    
}

/* Entry point for the software program.
* @argc    command line passed argument count and that the number of parameters passed
* @argv    command line passed argument values. This contains the images passed from the command line 
*/
int main(int argc, char** argv)
{
    ros::init(argc, argv, "camera");
    ROS_INFO("Camera node is running");
    ros::NodeHandle node;
    image_transport::ImageTransport transport(node);
    image_transport::Publisher img_publisher = transport.advertise("camera/image", 1);
    ros::Publisher int_publisher = node.advertise<imagineer::Number>("camera/integer", 1);
    std::string path = argv[1];
    std::vector<std::string> directory_files = get_files(path);
    ros::Rate loop(5000);

    while (node.ok())
    {    
        std::string image_file = pick_file(directory_files);
        Image image_to_publish = read_image(image_file);
        if (img_publisher.getNumSubscribers() > 0 && int_publisher.getNumSubscribers() > 0)
        {
            publish_message(node, img_publisher, int_publisher, image_to_publish);
        }
        else
        { 
            continue;
        }
        ros::spinOnce();
        loop.sleep();
    }
}