cmake_minimum_required(VERSION 2.8.12)

PROJECT(project1)
SET(CMAKE_CXX_STANDARD 14)

if( UNIX )
include_directories(~/anaconda2/include/python2.7 ~/anaconda2/lib/python2.7/site-packages/numpy/core/include/ ~/anaconda2/include)
endif()

#SET(OpenCV_DIR /home/hp/workfolder/OpenCV-Installation/installation/OpenCV-master/lib/cmake/opencv4)


######################## EDIT IF REQUIRED ####################
# ###Uncomment the line below and specify the path to OpenCV directory i.e. the path to the OpenCVConfig.cmake file. Check the examples given below.
#SET(OpenCV_DIR Enter-the-path-of-OpenCV-installation-on-your-system)


################### OpenCV_DIR Examples  #####################

### MACOS : /usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/

### UBUNTU : /usr/local/share/OpenCV/

### WINDOWS : C:\Users\yourname\Documents\opencv-3.3.1\build\install

##############################################################




###################   ***DO NOT EDIT***  #####################

#############  Common Instructions for all Users  ############
if(MSVC)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_app name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} )
ENDMACRO()

add_app(instagramFilters)
