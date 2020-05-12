# TMAT3004-Bacheloroppgave

trainer is based on RetinaNet and detectron2 https://detectron2.readthedocs.io/tutorials/install.html#common-installation-issues

App is written in C++. I recommend QtCreator open-source, and msvc++ from Microsoft Visual Studio Community Edition.

QtCreator can open the cmake file CMakeLists.txt

You need OpenCV, see https://medium.com/beesightsoft/build-opencv-opencv-contrib-on-windows-2e3b1ca96955 

Please use OpenCV 3.4, it has YOLOv4 support.

Collect your own dataset and put it into this directory, call the folder data. The structure is described in the report.

## Project overview

app consists of several projects

OBJECT_DETECTOR is the important project, it loads the model found in the data to count cod and saithe.

DATASET_TOOL is for improving the dataset, to create new labels

See Releases for executables of the project. You will find Windows and MacOS binaries. The Windows binaries include pre-compiled OpenCV libraries.

### RetinaNet
train.py will train a model with RetinaNet

inference.py will create a video with the RetinaNet model

### Report
report consists of my bachelor thesis
