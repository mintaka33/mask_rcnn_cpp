#include "segment.h"
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{
    FrameSegment fs;

    fs.init();

    int size = 1280 * 720;
    char *mask = new char[size];
    memset(mask, 0, size);
    fs.segment(mask);
    ofstream ofs;
    ofs.open("out.yuv", ios::binary);
    ofs.write(mask, size);
    ofs.close();

    fs.destroy();

    delete[] mask;
    return 0;
}
