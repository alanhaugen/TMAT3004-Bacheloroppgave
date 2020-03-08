cmake_minimum_required(VERSION 2.8.12)

PROJECT(app)
SET(CMAKE_CXX_STANDARD 14)

SET(Python2_ROOT_DIR ${PYTHONHOME})

find_package(Python2 COMPONENTS Development NumPy)

if( UNIX )
    include_directories(~/anaconda2/include/python2.7 ~/anaconda2/lib/python2.7/site-packages/numpy/core/include/ ~/anaconda2/include)
else()
    include_directories(${PYTHON_INCLUDE_DIRS} C:/Users/alanh/Anaconda2/Lib/site-packages/numpy/core/include)
endif()

#SET(OpenCV_DIR /c/opencv-4.1.0)

#############  Common Instructions for all Users  ############
if(MSVC)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

MACRO(add_app name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${PYTHON2_INCLUDE_DIRS} ${PYTHON2_NUMPY_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} Python2::Python )
ENDMACRO()

add_app(main)
