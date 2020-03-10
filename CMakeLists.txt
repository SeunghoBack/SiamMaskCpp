cmake_minimum_required(VERSION 2.8.3)
project(siammask)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

if (NOT DEFINED TORCH_PATH)
  execute_process(COMMAND python3 -c "import torch; print(torch.__path__[0], end='')" OUTPUT_VARIABLE TORCH_PATH RESULT_VARIABLE TORCH_FOUND)
  if (TORCH_FOUND EQUAL 0)
    message(STATUS "Configured TORCH_PATH: ${TORCH_PATH}")
  else()
    message(FATAL_ERROR "Failed to configure torch path automatically. Please specify option -DTORCH_PATH='/path/to/lib/python-3.x/site-packages/torch'")
  endif()
endif()

#set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  camera_info_manager
  sensor_msgs
  nodelet
  dynamic_reconfigure
  roslib
  niv_comm)

find_package(Torch REQUIRED PATHS ${TORCH_PATH})
find_package(OpenCV REQUIRED NO_DEFAULT_PATH PATHS /usr/local
  COMPONENTS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_highgui
  opencv_cudaimgproc
  opencv_cudawarping
)
find_package(CUDA REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

include_directories(
    "."
    "./src/third_party/argparse"
    "./src/third_party/json/include"
    "${TORCH_PATH}/include"
)

add_executable(demo src/demo.cpp)
target_link_libraries(demo ${catkin_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

add_executable(car_demo src/car_demo.cpp)
target_link_libraries(car_demo ${catkin_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

