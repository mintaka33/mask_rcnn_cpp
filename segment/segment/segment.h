#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>

using namespace std;

#ifdef MAKE_DLL
#  define DLLEXPORT __declspec(dllexport)
#else
#  define DLLEXPORT __declspec(dllimport)
#endif

class DLLEXPORT FrameSegment
{
public:
    FrameSegment();
    ~FrameSegment();

    int init();
    int init(string model, string config);
    int segment(char* nv12Buf, int width, int height, char* maskBuf);
    int destroy();
private:
    int loadNet();

private:
    float confThreshold = (float) 0.9; // Confidence threshold
    float maskThreshold = (float) 0.3; // Mask threshold
    string model = "";
    string config = "";
    string framework = "";
    int backendId = 0;
    int targetId = 0;
    float scale = (float)1.0; // (1/255)
    bool swapRB = true;
};

