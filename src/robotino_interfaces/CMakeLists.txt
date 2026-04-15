cmake_minimum_required(VERSION 3.8)
project(robotino_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/YoloDetect.srv"
  "srv/LocationsServer.srv" 
  "srv/FaceRecog.srv"
  "srv/Talk.srv"
  "srv/SetGrammar.srv"
  "srv/PoseDetect.srv"
  "msg/PersonTracking.msg"

  DEPENDENCIES vision_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
