cmake_minimum_required(VERSION 3.0.2)
project(lingua_mechanica_kinematics_server)

find_package(catkin REQUIRED COMPONENTS
    lingua_mechanica_kinematics_msgs rospy dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
    cfg/lingua_mechanica.cfg
)

catkin_package(
   CATKIN_DEPENDS lingua_mechanica_kinematics_msgs rospy dynamic_reconfigure
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
)


install(DIRECTORY launch scripts DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
