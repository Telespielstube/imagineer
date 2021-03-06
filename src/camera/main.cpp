#include <ros/ros.h>
#include <vector>
#include <time.h>
#include <experimental/filesystem>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <std_msgs/Int32.h>
#include "imagineer/Number.h"
#include "image.h"

/* Reads all available files from the directory into a list (vector).
* @path    Path to directory as an argument from the command line.
* @return  list (vector) of all files.
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

/* Function to pick a file randomly from the files list(vector). The picked file gets published.
*  @files     list of all files in the given directory.
*  @return    the randomly choosen file from the vector.
*/
std::string pick_file(std::vector<std::string> files)
{
    int random_file = (std::rand() % 10);
       
    return files.at(random_file);
}

/* Reads the content of the given file and saves content and filename as an unordered map.
* @image_files    a list of all files from the given command line path.
* @return         an Image object that hold the filename and the image as attributes.
*/
Image read_image(std::string image_file) 
{
    Image message;
    //int filename = std::stoi(image_file.substr(46, 47));
    int filename = std::stoi(std::experimental::filesystem::path(image_file).stem());
    ROS_INFO("File: %i", filename);
    cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
    message.set_filename(filename);
    message.set_image(image);
    return message;
}

/* Publishes the key, value pair as std_msgs and imagineer::Number messages to all subscribers.
* @node                  camera node object.
* @img_publisher         image transport publisher object.
* @int_message           integer publisher object.
# @message_to_publish    Image object to be sent.
*/
void publish_message(ros::NodeHandle node, image_transport::Publisher img_publisher, ros::Publisher int_publisher, 
                    Image message_to_publish)
{
    imagineer::Number int_message;
    int_message.digit = message_to_publish.get_filename();
    //ROS_INFO("Integer sent: %i", message_to_publish.get_name());
    int_publisher.publish(int_message);
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
    srand( (unsigned)time(NULL) );
    ros::NodeHandle node;
    image_transport::ImageTransport transport(node);
    image_transport::Publisher img_publisher = transport.advertise("camera/image", 1);
    ros::Publisher int_publisher = node.advertise<imagineer::Number>("camera/integer", 1);
    
    // the actual camera work is done here.
    std::string path = argv[1];
    std::vector<std::string> directory_files = get_files(path);
    ros::Rate loop(0.2);
    while (node.ok())
    {    
        if (img_publisher.getNumSubscribers() > 0 && int_publisher.getNumSubscribers() > 0)
        {
            std::string image_file = pick_file(directory_files);
            Image image_to_publish = read_image(image_file);
            publish_message(node, img_publisher, int_publisher, image_to_publish);
        }
        else
        { 
            continue;
        }
        loop.sleep();
        ros::spinOnce();
    }
}
