#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
//#include <opencv2/tracking.hpp>
#include <fstream>
//#include "matplotlibcpp.h"
#include <chrono>
#include <ctime>

using namespace std;
//using namespace matplotlibcpp;
using namespace cv::dnn;
using namespace cv;

// globals
float objectnessThreshold; // Objectness threshold
float confThreshold; // Confidence threshold
float nmsThreshold;  // Non-maximum suppression threshold
int inpWidth;  // Width of network's input image
int inpHeight; // Height of network's input image
vector<string> classes;

vector<Rect2d> bboxes;
int codQuantity, saitheQuantity;

string objectLabel;

const char *WINDOW_TITLE = "Press ESC to quit";

//bool isTracking;

//vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
//vector<Ptr<Tracker>> trackers;

// Create tracker by name
/*Ptr<Tracker> createTrackerByName(string trackerType)
{
    Ptr<Tracker> tracker;
    if (trackerType ==  trackerTypes[0])
        tracker = TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        tracker = TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = TrackerCSRT::create();
    else
    {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
        {
            std::cout << " " << *it << endl;
        }
    }

    return tracker;
}*/

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
    string label = format("%.1f", conf * 100);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + " " + label + "\%";
    }

    // Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

    objectLabel = label;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    codQuantity = 0;
    saitheQuantity = 0;

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

            if (confidence > confThreshold)
            {
                if (classes[classIdPoint.x] == "atlantic cod") {
                    codQuantity++;
                }
                else if (classes[classIdPoint.x] == "saithe")
                {
                    saitheQuantity++;
                }

                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));

                /*if (isTracking == false)
                {
                    Rect2d bbox(centerX - (width / 2), centerY - (height / 2), width, height); // KCF works best with a tight crop
                    bboxes.push_back(bbox);

                    Ptr<Tracker> newTracker = createTrackerByName(trackerTypes[2]); // Use KCF
                    newTracker->init(frame, Rect2d(bboxes[i]));

                    trackers.push_back(newTracker);
                }*/
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

    //isTracking = true;
}

int main(int argumentQuantity, char *arguments[])
{
    //isTracking = true;

    // Configuration for log file
    string filename = "log.csv";
    bool newFile = true;

    // Check if log file exists
    ifstream ifile(filename);
    if (ifile)
    {
        newFile = false;
    }

    // Open log file
    ofstream logFile;
    logFile.open(filename, std::ios_base::app);

    // Write what you will find in the log file on the first line if it does not already exist
    if (newFile)
    {
        logFile << "atlantic_cod_quantity,saithe_quantity,datetime" << endl;
    }

    // Open video file
    VideoCapture video;

    // Print usage as a warning
    clog << "Usage: ./OBJECT_DETECTOR <videoPath>" << endl;

    // Use webcam or filepath from command line
    if (argumentQuantity > 1)
    {
        string videoPath = arguments[1];
        video = VideoCapture(videoPath);
    }
    else
    {
        clog << "No video path given. Using camera 0" << endl;
        video = VideoCapture(0);
    }

    if (video.isOpened() == false)
    {
        cerr << "Failed to open video stream" << endl;
        return -1;
    }

    // Write out video with bounding boxes and labels to disk
    //VideoWriter videoWrite("out.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(width, height));

    // Frame matrix
    Mat frame;

    // Initialize the parameters
    int inpWidth = 416;  // Width of network's input image
    int inpHeight = 416; // Height of network's input image

    // Give the configuration and weight files for the model
    String modelConfiguration = "data/models/yolov3.cfg";
    String modelWeights = "data/models/yolov3.weights";

    // Check if model exists in data folder
    ifile = ifstream(modelConfiguration);
    if (!ifile)
    {
        cerr << "YOLOv3 model could not be found. Please download and extract data from here: https://www.dropbox.com/s/aym2lmnzjlam16v/data.zip" << endl;
        return -1;
    }

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    // Load names of classes
    string classesFile = "data/models/obj.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Configure user interface
    namedWindow(WINDOW_TITLE, WINDOW_AUTOSIZE);

    // Main loop
    bool isAlive = true;
    const int ESCAPE_KEYCODE = 27;
    const int DELAY_MILLISECONDS = 1;

    int key;

    while(isAlive)
    {
        key = waitKey(DELAY_MILLISECONDS);

        if (key == ESCAPE_KEYCODE)
        {
            isAlive = false;
            break;
        }

        /*if (key == 32)
        {
            isTracking = false;
        }*/

        double timer = (double)getTickCount();

        video >> frame;

        if (frame.empty())
        {
            isAlive = false;
            break;
        }

        // Loop through trackers manually
        /*for(int i = 0; i < int(bboxes.size()); i++)
        {
            Ptr<Tracker> tracker = trackers[i];
            Rect2d bbox = bboxes[i];

            // Update the tracking result with new frame
            bool ok = tracker->update(frame, bbox);

            if (ok)
            {
                // Draw tracked objects
                rectangle(frame, bbox, Scalar(128, 255, 50), 2, 1);
            }
            else
            {
                // Remove this tracker and bbox
                //trackers.erase(trackers.begin() + i);
                //bboxes.erase(bboxes.begin() + i);
            }
        }*/

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

        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        putText(frame, "FPS: " + std::to_string(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        putText(frame, "Atlantic cod quantity: " + std::to_string(codQuantity), Point(100,100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        putText(frame, "Saithe quantity: " + std::to_string(saitheQuantity), Point(100,130), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        //videoWrite.write(frame);
        imshow(WINDOW_TITLE, frame);

        // Get the time
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);

        // Remove the newline after time which is typically included from the C library call
        char *time = ctime(&end_time);
        if (time[strlen(time)-1] == '\n') time[strlen(time)-1] = '\0';

        // Log cod and saith quantities to csv file
        if (codQuantity > 0 || saitheQuantity > 0)
        {
            logFile << codQuantity << "," << saitheQuantity << ",\"" << time << "\"" << endl;
            cout << codQuantity << "," << saitheQuantity << ",\"" << time << "\"" << endl;
        }
    }

    logFile.close();

    video.release();
    //videoWrite.release();

    destroyAllWindows();

    return 0;
}
