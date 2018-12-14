# mask_rcnn_cpp
Mask-RCNN segmentation in C++

code sample for using segment library
```c++
    int width = 1280;
    int height = 720;

    // prepare input data, need nv12 format
    int szBufNV12 = width * height * 3 / 2;
    char *bufNV12 = new char[szBufNV12];
    ifstream ifs;
    ifs.open("test.yuv", ios::binary);
    ifs.read(bufNV12, szBufNV12);
    ifs.close();

    // create mask buffer
    int szBufMask = width * height;
    char *bufMask = new char[szBufMask];
    memset(bufMask, 0, szBufMask);

    // create a segment class & init
    FrameSegment seg;
    seg.init();

    // do Mask-RCNN segmentation, write result into mask buffer
    seg.segment(bufNV12, width, height, bufMask);

    // save mask buffer to file
    ofstream ofs;
    ofs.open("out.yuv", ios::binary);
    ofs.write(bufMask, szBufMask);
    ofs.close();

    // free resources
    seg.destroy();
    delete[] bufMask;
    delete[] bufNV12;
```
