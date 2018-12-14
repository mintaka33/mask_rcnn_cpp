#include "segment.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;

Net net;

float confThreshold = 0.9; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

vector<string> classes = { 
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
    "bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis",
    "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","","dining table","","","toilet",
    "","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};
vector<Scalar> colors = {
   {0.0, 255.0, 0.0, 255.0},
   {0.0, 0.0, 255.0, 255.0},
   {255.0, 0.0, 0.0, 255.0},
   {0.0, 255.0, 255.0, 255.0},
   {255.0, 255.0, 0.0, 255.0},
   {255.0, 0.0, 255.0, 255.0},
   {80.0, 70.0, 180.0, 255.0},
   {250.0, 80.0, 190.0, 255.0},
   {245.0, 145.0, 50.0, 255.0},
   {70.0, 150.0, 250.0, 255.0},
   {50.0, 190.0, 190.0, 255.0},
}; 

void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

    Scalar color = colors[classId%colors.size()];

    // Resize the mask, threshold, color and apply it on the image
    resize(objectMask, objectMask, Size(box.width, box.height));
    Mat mask = (objectMask > maskThreshold);
    Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    // Draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
    coloredRoi.copyTo(frame(box), mask);
}

void postprocess(Mat& frame, const std::vector<Mat>& outs)
{
    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; ++i)
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            // Extract the bounding box
            int classId = static_cast<int>(outDetections.at<float>(i, 1));
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

            left = max(0, min(left, frame.cols - 1));
            top = max(0, min(top, frame.rows - 1));
            right = max(0, min(right, frame.cols - 1));
            bottom = max(0, min(bottom, frame.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

            // Extract the mask for the object
            Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

            // Draw bounding box, colorize and show the mask on the image
            drawBox(frame, classId, score, box, objectMask);
        }
    }
}

void cvtNV12toYV12(char* yv12Buf, char* nv12Buf, int w, int h)
{
    int yv12Size = w * h * 3 / 2;
    char* srcY = nv12Buf;
    char* srcUV = nv12Buf + w * h;

    // Copy Y plane
    int ySize = w * h;
    memcpy_s(yv12Buf, ySize, srcY, ySize);

    // Copy U/V planes
    int uWidth = w / 2;
    int uHeight = h / 2;
    char* dstU = yv12Buf + ySize;
    char* dstV = dstU + uWidth * uHeight;
    for (int i = 0; i < h / 2; i++)
    {
        for (int j = 0; j < w / 2; j++)
        {
            dstV[i*uWidth + j] = srcUV[i*uWidth * 2 + j * 2];
            dstU[i*uWidth + j] = srcUV[i*uWidth * 2 + j * 2 + 1];
        }
    }
}

FrameSegment::FrameSegment()
{
}

FrameSegment::~FrameSegment()
{
}

int FrameSegment::init(string m, string c)
{
    model = m;
    config = c;
    loadNet();

    return 0;
}

int FrameSegment::init()
{
    model = "frozen_inference_graph.pb";
    config = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    loadNet();

    return 0;
}

int FrameSegment::segment(char* nv12Buf, int width, int height, char* maskBuf)
{
    Scalar mean = { 0.0, 0.0, 0.0, 0.0 };
    Mat frame, blob;

    //VideoCapture cap;
    //cap.open("test.mp4");
    //cap >> frame;

    char* bufYV12;
    int size = width * height * 3 / 2;
    bufYV12 = new char[size];
    memset(bufYV12, 0, size);
    cvtNV12toYV12(bufYV12, nv12Buf, width, height);
    frame = Mat(height * 3 / 2, width, CV_8UC1, bufYV12);
    if (frame.empty())
    {
        delete [] bufYV12;
        return -1;
    }

    blobFromImage(frame, blob, scale, Size(frame.cols, frame.rows), Scalar(), swapRB, false);
    net.setInput(blob);

    std::vector<String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";
    std::vector<Mat> outs;

    net.forward(outs, outNames);

    //postprocess(frame, outs);

    Mat outDetections = outs[0];
    Mat outMasks = outs[1];
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];
    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; ++i)
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            // Extract the bounding box
            int classId = static_cast<int>(outDetections.at<float>(i, 1));
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

            left = max(0, min(left, frame.cols - 1));
            top = max(0, min(top, frame.rows - 1));
            right = max(0, min(right, frame.cols - 1));
            bottom = max(0, min(bottom, frame.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

            // Extract the mask for the object
            Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

            // Resize the mask, threshold, color and apply it on the image
            resize(objectMask, objectMask, Size(box.width, box.height));
            Mat mask = (objectMask > maskThreshold);

            for (int y=top; y < bottom; y++)
            {
                for (int x= left; x < right; x++)
                {
                    maskBuf[y*frame.cols + x] = mask.at<char>(y-top, x-left);
                }
            }

            break;
        }
    }

    delete[] bufYV12;
    return 0;
}

int FrameSegment::destroy()
{
    return 0;
}

int FrameSegment::loadNet()
{
    net = readNet(model, config, framework);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    
    return 0;
}

