#ifndef __CANVAS_H
#define __CANVAS_H

#include "cuda_img.h"
#include "cuda_rect.h"
#include <iostream>

class Canvas
{
    public:
        const char* name;
        cv::Mat cv_canvas;
        uint2 size;
        CudaImg bg_image;
        CudaImg canvas;

        Canvas(const char* name, cv::Mat& cv_background, uint2 canvas_size);
        ~Canvas();
        void draw(CudaRect& rect); // draw a CudaRect to the canvas CudaImg
        void flush(); // inserts the default background CudaImg
        void show();
};

#endif//__CANVAS_H