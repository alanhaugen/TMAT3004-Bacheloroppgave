#include <opencv2/opencv.hpp>
#include "matplotlibcpp.h"

using namespace std;
using namespace matplotlibcpp;
using namespace cv;

int main()
{
    string videoPath = "data/video/soccer-ball.mp4";

    VideoCapture video(videoPath);

    if (video.isOpened() == false)
    {
        cout << "Failed to open video stream" << endl;
        return -1;
    }

    namedWindow("TMAT3004", WINDOW_AUTOSIZE);

    Mat frame;

    while(char c = waitKey(25) != 27)
    {
        video >> frame;

        if (frame.empty())
        {
            video.set(CAP_PROP_POS_FRAMES, 1);
            video >> frame;
        }

        imshow("TMAT3004", frame);
    }

    video.release();

    destroyAllWindows();
}
