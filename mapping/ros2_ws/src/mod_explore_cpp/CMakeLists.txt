cmake_minimum_required(VERSION 3.8)
project(mod_explore_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Eigen3 (header-only)
find_package(Eigen3 REQUIRED NO_MODULE)
if(NOT EIGEN3_INCLUDE_DIRS AND EIGEN3_INCLUDE_DIR)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})


add_executable(lio_post
  src/lio_post.cpp
)

ament_target_dependencies(lio_post
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  mavros_msgs
  std_msgs
  visualization_msgs
)

install(TARGETS
  lio_post
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
