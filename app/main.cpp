#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <fstream>
//#include "matplotlibcpp.h"

using namespace std;
//using namespace matplotlibcpp;
using namespace cv::dnn;
using namespace cv;

// YOLOv3 globals
float objectnessThreshold; // Objectness threshold
float confThreshold; // Confidence threshold
float nmsThreshold;  // Non-maximum suppression threshold
int inpWidth;  // Width of network's input image
int inpHeight; // Height of network's input image
vector<string> classes;

const int trackingFailureFramesQuantity = 10;

Rect2d bbox;

enum
{
    DETECTION,
    TRACKING
};

bool state;
bool isTrackerInitialized;

const char *WINDOW_TITLE = "Press ESC to quit";

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

// Get the names of the output layers
auto getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        // Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        // Get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    // Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    // Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    // Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (classes[classIdPoint.x] != "sports ball")
            {
                continue;
            }
            else if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));

                state = TRACKING;
                bbox = Rect2d(centerX - (width / 2), centerY - (height / 2), width, height);
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

int main()
{
//    string videoPath = "data/video/soccer-ball.mp4";
    string videoPath = "data/video/syntetisk_torsk.mkv";

    VideoCapture video(videoPath);
    //VideoCapture video(0);

    if (video.isOpened() == false)
    {
        cout << "Failed to open video stream" << endl;
        return -1;
    }

    namedWindow(WINDOW_TITLE, WINDOW_AUTOSIZE);

    Mat frame;

    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "CSRT", "MOSSE"};

    string trackerType = trackerTypes[2];

    Ptr<Tracker> tracker;

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

    state = DETECTION;
    isTrackerInitialized = false;

    bool isDetectionRecent = true;
    int frames = 0;

    video >> frame;

    // Initialize the parameters
    float objectnessThreshold = 0.5; // Objectness threshold
    float confThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    int inpWidth = 416;  // Width of network's input image
    int inpHeight = 416; // Height of network's input image

    // Load names of classes
    string classesFile = "data/models/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "data/models/yolov3.cfg";
    String modelWeights = "data/models/yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    while(char c = waitKey(25) != 27)
    {
        double timer = (double)getTickCount();

        video >> frame;

        if (frame.empty())
        {
            video.set(CAP_PROP_POS_FRAMES, 1);
            video >> frame;
        }

        if (state == DETECTION)
        {
            putText(frame, "DETECTION", Point(100,70), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
            putText(frame, "Tracking failure detected", Point(100,90), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);

            // Create a 4D blob from a frame.
            Mat blob;
            blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

            //Sets the input to the network
            net.setInput(blob);

            // Runs the forward pass to get output of the output layers
            vector<Mat> outs;
            net.forward(outs, getOutputsNames(net));

            // Remove the bounding boxes with low confidence
            postprocess(frame, outs);

            // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            string label = format("Inference time for a frame : %.2f ms", t);
            putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        }
        else if (state == TRACKING)
        {
            if (isTrackerInitialized == false)
            {
                tracker->init(frame, bbox);
                isTrackerInitialized = true;
            }

            putText(frame, "TRACKING", Point(100,70), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

            bool ok = tracker->update(frame, bbox);

            if (ok)
            {
                // Tracking success, draw green rectangle around object
                rectangle(frame, bbox, Scalar(0, 255, 0), 2, 1 );
                isDetectionRecent = false;
            }
            else
            {
                frames++;
                if (frames > trackingFailureFramesQuantity || isDetectionRecent == true)
                {
                    frames = 0;
                    state = DETECTION;
                    isTrackerInitialized = false;
                    isDetectionRecent = true;
                }
            }
        }

        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        putText(frame, trackerType + " Tracker", Point(100,150), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

        imshow(WINDOW_TITLE, frame);
    }

    video.release();

    destroyAllWindows();
}
