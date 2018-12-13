#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class FrameSegment
{
public:
    FrameSegment();
    ~FrameSegment();

    int init();
    int init(string model, string config);
    int segment();
    int destroy();
private:
    int loadNet();

private:
    float confThreshold = (float) 0.9; // Confidence threshold
    float maskThreshold = (float) 0.3; // Mask threshold
    string model;
    string config;
    string framework = "";
    int backendId = 0;
    int targetId = 0;
    float scale = (float)1.0; // (1/255)
    Scalar mean = { 0.0, 0.0, 0.0, 0.0 };
    bool swapRB = true;
};

