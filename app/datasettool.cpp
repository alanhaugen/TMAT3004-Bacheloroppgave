#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
//using namespace matplotlibcpp;
using namespace cv;

const char *WINDOW_TITLE = "Press ESC to quit";

int rectX, rectY;
Mat frame, backbuffer;

int imageQuantity;

std::ofstream fishTrainFile, labelFile;

enum classes
{
    ATLANTIC_COD = 0,
    SAITHE
};

static void onMouse(int event, int x, int y, int flags, void* param)
{
    switch(event)
    {
        case EVENT_LBUTTONDOWN:
            rectX = x;
            rectY = y;

            break;
        case EVENT_LBUTTONUP:
            int className = classes::ATLANTIC_COD;
            string data = to_string(className) + " " + to_string(rectX) + " " + to_string(rectY) + " " + to_string(x) + " " + to_string(y);

            labelFile << data << endl;

            Rect rect(rectX, rectY, (x-rectX), (y-rectY));
            rectangle(frame, rect, Scalar(0,255,0), 5);

            break;
    }
}

void saveImage()
{
    if (labelFile.is_open())
    {
        labelFile.close();
    }

    string labelPath = "data/labels/fish_" + to_string(imageQuantity) + ".txt";
    labelFile.open(labelPath, std::ios_base::app);

    string filepath = "data/train/atlantic_cod/fish_" + to_string(imageQuantity) + ".png";
    fishTrainFile << filepath << endl;

    imwrite(filepath, backbuffer);

    imageQuantity++;
}

int main()
{
    imageQuantity = 590;

    string videoPath = "in.mp4";

    VideoCapture video(videoPath);

    if (video.isOpened() == false)
    {
        cout << "Failed to open video stream" << endl;
        return -1;
    }

    namedWindow(WINDOW_TITLE, WINDOW_AUTOSIZE);

    setMouseCallback(WINDOW_TITLE, onMouse);

    video.set(CAP_PROP_POS_FRAMES, imageQuantity);

    video >> frame;
    backbuffer = frame.clone();

    fishTrainFile.open("data/fish_train.txt", std::ios_base::app);

    saveImage();

    bool isAlive = true;

    while(isAlive)
    {
        int key = waitKey(25);

        if (key == 27)
        {
            isAlive = false;
        }
        else if (key == 32)
        {
            video >> frame;
            backbuffer = frame.clone();

            saveImage();
        }

        putText(frame, "Press SPACE to continue to next frame", Point(100,90), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        putText(frame, to_string(imageQuantity) + " frame", Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        imshow(WINDOW_TITLE, frame);
    }

    video.release();
    destroyAllWindows();

    labelFile.close();
    fishTrainFile.close();
}
