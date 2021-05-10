#include <ros/ros.h>
#include <vector>
#include <unordered_map>
#include <experimental/filesystem>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <std_msgs/Int32.h>
#include "imagineer/Number.h"

class Camera
{
    public:
        Camera() {}

        Camera() : transport(node)
        {
            img_publisher = transport.advertise("camera/image", 1);
            int_publisher = node.advertise<imagineer::Number>("camera/integer", 1);
            path = argv[1];
            directory_files = get_folder_content(path);
            message_list = read_image(directory_files);
            publish_message(node, img_publisher, int_publisher, message_list);    
        }

        /* Reads all available files from the directory.
        * @path    Path to directory as an argument from the command line.
        * @return  list of all files.
        */
        std::vector<std::string> get_folder_content(std::string path);

        /* Reads the content of the given file and saves content and filename as unorrdered map.
        * @image_files    a list of all files from the given command line path.
        * @return         a (key, value) data structure that hold the filename(key) and the image(value).
        */
        std::unordered_map<int, sensor_msgs::ImagePtr> read_image(std::vector<std::string> image_files);

        /* Publishes the key, value pair as std_msgs and imagineer::Number messages to all subscribers.
        * @node            Node object.
        * @img_publisher   image transport publisher object.
        * @int_message     integer publisher object.
        # @message_list    a (key, value) data structure that hold the filename(key) and the image(value).
        */
        void publish_message(ros::NodeHandle node, image_transport::Publisher img_publisher, ros::Publisher int_publisher, 
                            std::unordered_map<int, sensor_msgs::ImagePtr> message_list);
    
    private:
        ros::NodeHandle node;
        image_transport::ImageTransport transport;
        image_transport::Publisher img_publisher;
        ros::Publisher int_publisher;
        
        std::string path;
        std::vector<std::string> directory_files;
        std::unordered_map<int, sensor_msgs::ImagePtr> message_list;
};