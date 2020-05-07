cmake_minimum_required(VERSION 2.8.12)

PROJECT(app)
SET(CMAKE_CXX_STANDARD 14)

SET(Python2_ROOT_DIR ${PYTHONHOME})

find_package(Python2 COMPONENTS Development NumPy)

SET(OpenCV_DIR C:/opencv-3.4/opencv/build)

if( UNIX )
    include_directories(~/anaconda2/include/python2.7 ~/anaconda2/lib/python2.7/site-packages/numpy/core/include/ ~/anaconda2/include)
else()
    include_directories(${PYTHON_INCLUDE_DIRS} ~/Anaconda2/Lib/site-packages/numpy/core/include)
endif()

if(MSVC)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

find_package( OpenCV REQUIRED )
#find_package(Torch REQUIRED)

set(OpenCV_STATIC ON) # Easier to share binaries

include_directories( ${OpenCV_INCLUDE_DIRS} )

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(PROJECT_4 yolo.cpp)
target_include_directories(PROJECT_4 PRIVATE ${PYTHON2_INCLUDE_DIRS} ${PYTHON2_NUMPY_INCLUDE_DIRS})
target_link_libraries(PROJECT_4 PRIVATE ${OpenCV_LIBS} Python2::Python ) # libs # "${TORCH_LIBRARIES}"

add_executable(OBJECT_DETECTOR main.cpp)
target_include_directories(OBJECT_DETECTOR PRIVATE ${PYTHON2_INCLUDE_DIRS} ${PYTHON2_NUMPY_INCLUDE_DIRS})
target_link_libraries(OBJECT_DETECTOR PRIVATE ${OpenCV_LIBS} Python2::Python ) # libs # "${TORCH_LIBRARIES}"

add_executable(DATASET_TOOL datasettool.cpp)
target_include_directories(DATASET_TOOL PRIVATE ${PYTHON2_INCLUDE_DIRS} ${PYTHON2_NUMPY_INCLUDE_DIRS})
target_link_libraries(DATASET_TOOL PRIVATE ${OpenCV_LIBS} Python2::Python ) # libs # "${TORCH_LIBRARIES}"
