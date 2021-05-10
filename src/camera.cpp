#include <ros/ros.h>
#include <vector>
#include <unordered_map>
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
        Image(int filename, cv::Mat content) 
        {
            number = filename;
            img = content;
        }
    
    private:
        int name;
        cv::Mat img;
}
 
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
    return file;
}

/* Function to pick a file from the files vector. This file gets published.
*  @files     list of all files in the given directory.
*  @return    the randomly choosen file from the vector.
*/
std::string<std::string> pick_file(std::vector<std::string> files)
{
    int random_file = 5 + (std::rand() % (9 -  + 1));
    std::string pick = "";
    for (entry : files)
    {
        pick = entry.at(random_file);
    }
    return pick;
}

/* Reads the content of the given file and saves content and filename as an unordered map.
* @image_files    a list of all files from the given command line path.
* @return         a (key, value) data structure that hold the filename(key) and the image(value).
*/
Image read_image(std::string image_file)
{
    Image image_to_publish;
    int filename = 0;
    filename = std::stoi(entry.substr(16, 17));
    cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
    image_to_publish.number = filename;
    image_to_publish.img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();  
    return image_to_publish;
}

/* Publishes the key, value pair as std_msgs and imagineer::Number messages to all subscribers.
* @node            Node object.
* @img_publisher   image transport publisher object.
* @int_message     integer publisher object.
# @Image           Image object with filename and image as attributes.
*/
void publish_message(ros::NodeHandle node, image_transport::Publisher img_publisher, ros::Publisher int_publisher, 
                    Image message_list)
{
    if (img_publisher.getNumSubscribers() > 0 && int_publisher.getNumSubscribers() > 0)
    {
        imagineer::Number message;
        message.digit = entry.first;
        int_publisher.publish(message);
        img_publisher.publish(entry.second);       
    }
    else
    { 
        continue;
    }
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
    while (node.ok)
    {    
        std::vector<std::string> image_file = pick_file(directory_files);
        Image image_to_publish = read_image(image_file);
        publish_message(node, img_publisher, int_publisher, image_to_publish);
        ros::spinOnce();
        loop.sleep();
    }
}