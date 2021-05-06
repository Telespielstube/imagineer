#include <ros/ros.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <std_msgs/Int32.h>
#include "imagineer/Number.h"

// class Camera
// {
//     public:
    
// };

std::vector<std::string> get_folder_content(char** path)
{
    std::vector<std::string> files;
    const std::string _path(path);
    const std::string img_file;
    for (img_file : std::filesystem::directory_iterator(_path))
    {
        files.push_back(file);
    }
    return files;
}

/* Reads the content of the given file and saves content and filename as unorrdered map.
*/
std::unordered_map<char, sensor_msgs::ImagePtr> read_image(std::vector<char**> image_files)
{
    std::unordered_map<char, sensor_msgs::ImagePtr> = message_to_publish;
    //fills the unordered map with filename as key and image as value sensor_msgs.
    for (cv::Mat img : sensor_msgs::ImagePtr)
    {
        char filename = image_files.substr(0, 1);
        cv::Mat image = cv::imread(image_files, cv::IMREAD_COLOR);
        message_to_publish.insert(filename, cv_bridge::CvImage(std_msgs::Header(), "bgr8", image)).toImageMsg();   
    }
    return message_to_p ublish;
}

void publish_message(image_transport::Publisher img_publisher, ros::Publisher int_publisher, 
                    std:unordered_map<char, sensor_msgs::ImagePtr> message_list)
{
    ros::Rate loop(50);
    while (node.ok()) 
    {
        if (img_publisher.getNumSubscribers() > 0 && int_publisher.getNumSubscribers() > 0)
        {
            for (image : image_list)
            {
                int_publisher.publish((int)message_list->first));
                img_publisher.publish(message_list->second);
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
    std::vector<std::string> directory_files = get_folder_content(argv[1]);
    std:unordered_map<char, sensor_msgs::ImagePtr> message_list = read_image(directory_files);
    publish_message(img_publisher, int_publisher, message_list);
 
    // as long as the node is running and at least one node
    // subscribes to the two topics the camera node sends both messages.
    // while (node.ok()) 
    // {
    //     if (img_publisher.getNumSubscriber() > 0 && int_publisher.getNumSubscriber() > 0)
    //     {
    //         img_publisher.publish(img_message);
    //         int_publisher.publish(int_message);
    //     }
    //     else{ 
    //         continue;
    //     }
    //     ros::spinOnce();
    //     loop.sleep();
    // }
}