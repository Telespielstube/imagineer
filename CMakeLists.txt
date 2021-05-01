cmake_minimum_required(VERSION 2.8.3)
project(imagineer)

set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4/)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

## Find catkin and any catkin packages
find_package(
    catkin REQUIRED COMPONENTS
    OpenCV REQUIRED	
    sensor_msgs
    cv_bridge
    image_transport
    roscpp 
    rospy 
    std_msgs 
    message_generation
)

## Declare ROS messages and services
add_message_files(
    FILES 
    Image.msg
    Number.msg
)

add_service_files(
    FILES 
    ImageAck.srv
)

## Generate added messages and services
generate_messages(
    DEPENDENCIES 
    std_msgs
    sensor_msgs
)

## Declare a catkin package
catkin_package(
    CATKIN_DEPENDS message_runtime
)

# Build the neural_network service.
catkin_install_python(PROGRAMS scripts/ai_service.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Build camera, processor, controller.
include_directories(include ${catkin_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(camera src/camera.cpp)
target_link_libraries(camera ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} )
add_dependencies(camera imagineer_generate_messages_cpp)

add_executable(processor src/processor.cpp)
target_link_libraries(processor ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} )
add_dependencies(processor imagineer_generate_messages_cpp)

add_executable(controller src/controller.cpp)
target_link_libraries(controller ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} )
add_dependencies(controller imagineer_generate_messages_cpp)
