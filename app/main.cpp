#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
//#include "matplotlibcpp.h"

using namespace std;
//using namespace matplotlibcpp;
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

    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "CSRT", "MOSSE"};

    string trackerType = trackerTypes[6];

    Ptr<Tracker> tracker;
    Rect2d bbox(204, 131, 97, 222);

    if (trackerType == "BOOSTING")
        tracker = TrackerBoosting::create();
    else if (trackerType == "MIL")
        tracker = TrackerMIL::create();
    else if (trackerType == "KCF")
        tracker = TrackerKCF::create();
    else if (trackerType == "TLD")
        tracker = TrackerTLD::create();
    else if (trackerType == "MEDIANFLOW")
        tracker = TrackerMedianFlow::create();
    else if (trackerType == "GOTURN")
        tracker = TrackerGOTURN::create();
    else if (trackerType == "CSRT")
        tracker = TrackerCSRT::create();
    else if (trackerType == "MOSSE")
        tracker = TrackerMOSSE::create();

    while(char c = waitKey(25) != 27)
    {
        video >> frame;

        if (frame.empty())
        {
            video.set(CAP_PROP_POS_FRAMES, 1);
            video >> frame;
        }

        bool ok = tracker->update(frame, bbox);

        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }

        imshow("TMAT3004", frame);
    }

    video.release();

    destroyAllWindows();
}
