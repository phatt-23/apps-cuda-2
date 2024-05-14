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

        /// @brief # Canvas comstructor
        /// @param name window name when show() method is called
        /// @param cv_background cv::Mat& representing an image 
        /// @param canvas_size {y,x}
        /// @param flags CV_8UC3 (for RGB canvas) or CV_8UC4 (for RGBA canvas) 
        Canvas(const char* name, cv::Mat& cv_background, uint2 canvas_size, size_t flags = CV_8UC3);

        ~Canvas();
        void draw(CudaRect& rect); // draw a CudaRect to the canvas CudaImg
        void flush(); // inserts the default background CudaImg
        void show();
};

#endif//__CANVAS_H