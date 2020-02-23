#include <opencv2/opencv.hpp>
#include "matplotlibcpp.h"

using namespace std;
using namespace matplotlibcpp;
using namespace cv;

Mat cartoonify(Mat image, int arguments=0)
{
    Mat cartoonImage;

    /// YOUR CODE HERE

    return cartoonImage;
}

Mat pencilSketch(Mat image, int arguments=0)
{
    Mat pencilSketchImage;
    
    /// YOUR CODE HERE

    return pencilSketchImage;
}

int main()
{
    string imagePath = "images/trump.jpg";
    Mat image = imread(imagePath);

    Mat cartoonImage = cartoonify(image);
    Mat pencilSketchImage = pencilSketch(image);

    imshow("Original Image", image);
    //imshow("Pencil Sketch Result", cartoonImage);
    //imshow("Cartoon Filter Result", pencilSketchImage);

    waitKey(0);
}
