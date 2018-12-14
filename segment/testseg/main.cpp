#include "segment.h"
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{
    int width = 1280;
    int height = 720;

    // load nv12 data 
    int szBufNV12 = width * height * 3 / 2;
    char *bufNV12 = new char[szBufNV12];
    ifstream ifs;
    ifs.open("test.yuv", ios::binary);
    ifs.read(bufNV12, szBufNV12);
    ifs.close();

    // mask buffer
    int szBufMask = width * height;
    char *bufMask = new char[szBufMask];
    memset(bufMask, 0, szBufMask);

    FrameSegment fs;
    fs.init();

    // do Mask RCNN segmentation
    fs.segment(bufNV12, width, height, bufMask);

    // save mask buffer to file
    ofstream ofs;
    ofs.open("out.yuv", ios::binary);
    ofs.write(bufMask, szBufMask);
    ofs.close();

    fs.destroy();

    delete[] bufMask;
    delete[] bufNV12;
    return 0;
}
