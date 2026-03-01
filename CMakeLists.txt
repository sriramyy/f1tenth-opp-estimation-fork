cmake_minimum_required(VERSION 3.5)
project(lidar_processing)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =====================
# Find dependencies FIRST
# =====================
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(laser_geometry REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common kdtree segmentation filters)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# =====================
# Python package install
# =====================
ament_python_install_package(${PROJECT_NAME})

# =====================
# Includes
# =====================
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# =====================
# C++ executables
# =====================
add_executable(scan_to_cloud src/scan_to_cloud.cpp)
ament_target_dependencies(scan_to_cloud
  rclcpp
  sensor_msgs
  laser_geometry
  tf2
  tf2_ros
  tf2_sensor_msgs
)

add_executable(bounding_box src/bounding_box.cpp)
ament_target_dependencies(bounding_box
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
)


# =====================
# Install targets
# =====================
install(TARGETS
  scan_to_cloud
  bounding_box
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/imm_filterpy.py
  scripts/interceptor.py
  scripts/clustering.py
  scripts/global_raceline_publisher.py
  scripts/walls_gen.py
  DESTINATION lib/${PROJECT_NAME}
)


# =====================
# Testing
# =====================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
