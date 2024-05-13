#ifndef __CUDA_RECTANGLE_H
#define __CUDA_RECTANGLE_H

#include <iostream>
#include "cuda_img.h"

class CudaRect : public CudaImg
{
    public:
        float2 pos;
        uint alpha;
        uint2 rsize;
        cv::Mat* cv_mat_ptr;

        CudaRect(cv::Mat& img, int2 pos, uint alpha);
        CudaRect(cv::Mat& img, uint2 rsize, int2 pos, uint alpha);
        ~CudaRect();

        void set_pos(float y, float x); // sets the pos to bhave these values
        void add_pos(float y, float x); // adds these values to the pos
};

#endif//__CUDA_RECTANGLE_H

