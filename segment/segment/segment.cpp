#include "segment.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;


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

int FrameSegment::segment()
{
    Scalar mean = { 0.0, 0.0, 0.0, 0.0 };

    return 0;
}

int FrameSegment::destroy()
{
    return 0;
}

int FrameSegment::loadNet()
{
    Net net = readNet(model, config, framework);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    
    return 0;
}

