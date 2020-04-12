#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
//using namespace matplotlibcpp;
using namespace cv;

const char *WINDOW_TITLE = "Press ESC to quit";

int rectX, rectY;
Mat frame, backbuffer, prevFrame;

int imageQuantity, imageSkip;

string labelData;

std::ofstream fishTrainFile, labelFile;

bool saveData;

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
        case EVENT_RBUTTONDOWN:
            frame = prevFrame;
            saveData = false;

            break;
        case EVENT_LBUTTONUP:
            if (rectX > x)
            {
               int old = rectX;
               rectX = x;
               x = old;
            }

            if (rectY > y)
            {
                int old = rectY;
                rectY = y;
                y = old;
            }

            if (saveData) labelFile << labelData << endl;
            saveData = true;

            int className = classes::ATLANTIC_COD;
            labelData = to_string(className) + " " + to_string(rectX) + " " + to_string(rectY) + " " + to_string(x) + " " + to_string(y);

            prevFrame = frame.clone();

            Rect rect(rectX, rectY, (x-rectX), (y-rectY));
            rectangle(frame, rect, Scalar(0,255,0), 1);

            break;
    }
}

void plotTruth()
{
    ifstream fin;
    string line;
    string labelPath = "data/labels/fish_" + to_string(imageQuantity - imageSkip) + ".txt";
    // Open an existing file
    fin.open(labelPath);

    std::string::size_type sz;

    for (int i = 0; !fin.eof(); i++)
    {
        fin >> line;

        int classId, x1, y1, x2, y2;

        switch(i)
        {
            case 0:
                // Class (atlantic cod or saithe)
                classId = atoi(line.c_str());
                break;
            case 1:
                x1 = atoi(line.c_str());
                break;
            case 2:
                y1 = atoi(line.c_str());
                break;
            case 3:
                x2 = atoi(line.c_str());
                break;
            case 4:
                y2 = atoi(line.c_str());
                break;
            default:
                // Draw rect
                Rect rect(x1, y1, (x2-x1), (y2-y1));
                rectangle(frame, rect, Scalar(0,255,classId * 255), 1);

                // Loop back and start on next line
                i = 0;
        }
    }

    fin.close();
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
    fishTrainFile << "train/atlantic_cod/fish_" + to_string(imageQuantity) + ".png" << endl;

    imwrite(filepath, backbuffer);

    imageQuantity += imageSkip;
}

int main()
{
    imageQuantity = 590;
    imageSkip = 50;

    saveData = false;

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
    prevFrame = frame.clone();

    fishTrainFile.open("data/fish_train.txt", std::ios_base::app);

    saveImage();
    plotTruth();

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
            video.set(CAP_PROP_POS_FRAMES, imageQuantity + imageSkip);
            video >> frame;
            backbuffer = frame.clone();
            prevFrame = frame.clone();
            if (saveData) labelFile << labelData << endl;
            saveData = false;

            saveImage();
            plotTruth();
        }

        putText(frame, "Press SPACE to continue to next frame", Point(100,90), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        putText(frame, to_string(imageQuantity - imageSkip) + " frame", Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        imshow(WINDOW_TITLE, frame);
    }

    video.release();
    destroyAllWindows();

    if (saveData) labelFile << labelData << endl;

    labelFile.close();
    fishTrainFile.close();
}
