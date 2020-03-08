#include <opencv2/opencv.hpp>
#include "matplotlibcpp.h"

using namespace std;
using namespace matplotlibcpp;
using namespace cv;

int main()
{
    string imagePath = "data/images/opencv-1-cpp";
    Mat image = imread(imagePath);

    imshow("Image", image);

    waitKey(0);
}
