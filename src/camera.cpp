#include <ros/ros.h>
#include <vector>
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

std::vector<std::strirng> get_folder_content(char** path)
{
    std::vector<char*> files;
    string path = path;
    for (file : directory_iterator(path))
    {
        files.push_back(file);
    }
    return files;
}

/* Reduces the filename to one character.
* file_list    list of all files in the given direrctory.
* filenames    one character long filename 
*/
std::vector<std::string> get_filename(std::vector<charr**> file_list)
{
    std::vector<char> filenames;
    for (file : file_list)
    {
        filenames.push_back(file.substr(0, 1);
    }
    return filenames;
}

/* Reads the content ideally image content of the given file.
*/
std::vector<cv::Mat> read_image(std::vector<char**> image_files)
{
    std::vector<cv::Mat> images;
    for (cv::Mat img : img_files)
    {
        cv::Mat image = cv::imread(image_dir, cv::IMREAD_COLOR);
        images.push_back(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    }
    return images;
    //sensor_msgs::ImagePtr img_message = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
}

void publish_message(std::vector<cv::Mat> image_list, std::vector<std::string> filename_list)
{
    ros::Rate loop(50);
    while (node.ok()) 
    {
        if (img_publisher.getNumSubscriber() > 0 && int_publisher.getNumSubscriber() > 0)
        {
            for (image : image_list)
            {
                img_publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());
                int_publisher.publish((int)filename_list));
            }    
        }
        else{ 
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
    std::vector<std::string> filenames = get_filename(directory_files); 
    std::vector<cv::Mat> images = read_image(directory_files);
    publish_message(filenames, images);
    std_msgs::Header header;
    imagineer::Number int_message; 
    int_message.digit = 2;
    publish_message(img_publisher, int_publisher )

    
    // as long as the node is running and at least one node
    // subscribes to the two topics the camera node sends both messages.
    while (node.ok()) 
    {
        if (img_publisher.getNumSubscriber() > 0 && int_publisher.getNumSubscriber() > 0)
        {
            img_publisher.publish(img_message);
            int_publisher.publish(int_message);
        }
        else{ 
            continue;
        }
        ros::spinOnce();
        loop.sleep();
    }
}