# TMAT3004-Bacheloroppgave

trainer is based on RetinaNet and detectron2 https://detectron2.readthedocs.io/tutorials/install.html#common-installation-issues

App is written in C++. I recommend QtCreator open-source, and msvc++ from Microsoft Visual Studio Community Edition.

QtCreator can open the cmake file CMakeLists.txt

You need OpenCV, see https://medium.com/beesightsoft/build-opencv-opencv-contrib-on-windows-2e3b1ca96955 

Download data used for training here: https://www.dropbox.com/s/aym2lmnzjlam16v/data.zip

## Project overview

app consists of several projects

OBJECT_DETECTOR is the important project, it loads the model found in the data to count cod and saithe.

DATASET_TOOL is for improving the dataset, to create new labels

PROJECT_4 is just a test

See Releases for executables of the project. You will find Windows and MacOS binaries.

### RetinaNet
train.py will train a model with RetinaNet

inference.py will create a video with the RetinaNet model

Makefile will automatically download the data, and run train.py then inference.py

### Report
report consists of my bachelor thesis
