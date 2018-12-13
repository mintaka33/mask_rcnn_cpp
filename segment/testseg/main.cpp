#include "segment.h"

int main(int argc, char* argv[])
{
    FrameSegment fs;

    fs.init();
    fs.segment();
    fs.destroy();

    return 0;
}
