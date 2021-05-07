#include <ros/ros.h>
#include <vector>
#include <unordered_map>
#include <experimental/filesystem>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <std_msgs/Int32.h>
#include "imagineer/Number.h"

// class Camera
// {
//     public:
    
// };

/* Reads all available files from the directory.
* @path    Path to directory as an argument from the command line.
* @return  list of all files.
*/
std::vector<std::string> get_folder_content(std::string path)
{
    std::vector<std::string> files;
    std::experimental::filesystem::directory_iterator path_iterator(path);
    for (const auto& entry : path_iterator)
    {
        files.push_back(entry.path().string());
    }
    return files;
}

/* Reads the content of the given file and saves content and filename as unorrdered map.
* @image_files    a list of all files from the given command line path.
* @return         a (key, value) data structure that hold the filename(key) and the image(value).
*/
std::unordered_map<int, sensor_msgs::ImagePtr> read_image(std::vector<std::string> image_files)
{
    std::unordered_map<int, sensor_msgs::ImagePtr> message_to_publish;
    int filename = 0;
    //fills the unordered map with filename as key and image as value sensor_msgs.  
    for (std::string entry : image_files)
    {
        filename = std::stoi(entry.substr(16, 17));
        cv::Mat image = cv::imread(entry, cv::IMREAD_COLOR);
        message_to_publish[filename] = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg(); // adds filename as key and cv_bridge Image as value  
    }
    return message_to_publish;
}

/* Publishes the key, value pair as std_msgs and imagineer::Number messages to all subscribers.
* @node            Node object.
* @img_publisher   image transport publisher object.
* @int_message     integer publisher object.
# @message_list    a (key, value) data structure that hold the filename(key) and the image(value).
*/
void publish_message(ros::NodeHandle node, image_transport::Publisher img_publisher, ros::Publisher int_publisher, 
                    std::unordered_map<int, sensor_msgs::ImagePtr> message_list)
{
    ros::Rate loop(50);
    while (node.ok()) 
    {
        if (img_publisher.getNumSubscribers() > 0 && int_publisher.getNumSubscribers() > 0)
        {
            for (auto entry : message_list)
            {
                imagineer::Number message;
                message.digit = entry.first;
                int_publisher.publish(message);
                img_publisher.publish(entry.second);
            }    
        }
        else
        { 
            continue;
        }
        ros::spinOnce();
        loop.sleep();
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
    std::vector<std::string> directory_files = get_folder_content(path);
    std::unordered_map<int, sensor_msgs::ImagePtr> message_list = read_image(directory_files);
    publish_message(node, img_publisher, int_publisher, message_list);
}