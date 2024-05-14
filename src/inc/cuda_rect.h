#ifndef __CUDA_RECTANGLE_H
#define __CUDA_RECTANGLE_H

#include <iostream>
#include "cuda_img.h"

class CudaRect : public CudaImg
{
    public:
        cv::Mat* cv_mat_ptr;
        uint2 rsize;
        float2 pos;
        uint alpha;

        /// # CudaRect costructor - texture, position
        /// @attention Will use the cv::Mat& texture size as its own size (rsize)
        /// @attention and set the alpha to 255 (full opacity).
        /// @param texture cv::Mat& representing an image
        /// @param pos int2 initial position {y, x}
        CudaRect(cv::Mat& texture, int2 pos);
        
        /// # CudaRect costructor - texture, position, rect-size
        /// @attention Will set the alpha to 255 (full opacity).
        /// @param texture cv::Mat& representing an image
        /// @param pos int2 initial position {y, x}
        /// @param rsize uint2 rectangle's size      
        CudaRect(cv::Mat& texture, int2 pos, uint2 rsize);

        /// # CudaRect costructor - texture, position, alpha
        /// @attention Will use the cv::Mat& texture size as its own size (rsize).
        /// @param texture cv::Mat& representing some image
        /// @param pos int2 initial position {y, x}
        /// @param alpha uint controls how opaque the CudaRect is [0,255]
        CudaRect(cv::Mat& texture, int2 pos, uint alpha);

        /// # CudaRect costructor - texture, position, rect-size, alpha
        /// @param texture cv::Mat& representing some image
        /// @param pos int2 initial position {y, x}
        /// @param rsize uint2 rectangle's size  
        /// @param alpha uint controls how opaque the CudaRect is [0,255]
        CudaRect(cv::Mat& texture, int2 pos, uint2 rsize, uint alpha);

        ~CudaRect();

        void set_pos(float y, float x); // sets the pos to bhave these values
        void add_pos(float y, float x); // adds these values to the pos
};

#endif//__CUDA_RECTANGLE_H

