cmake_minimum_required(VERSION 2.8.3)
project(profi2022_master_solution)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# Debug option
ADD_DEFINITIONS(-DDEBUG)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  cv_bridge
  std_msgs
  image_transport
)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS
)

include_directories (
  ${catkin_INCLUDE_DIRS}
)

# For c++ example
add_executable(simple_move_node src/simple_move.cpp)
add_dependencies(simple_move_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(simple_move_node ${OpenCV_LIBS} ${catkin_LIBRARIES})

# For python example
catkin_install_python(PROGRAMS scripts/simple_move.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS
  simple_move_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
