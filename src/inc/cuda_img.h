#ifndef __CUDA_IMAGE_H
#define __CUDA_IMAGE_H

#include <opencv2/core/mat.hpp>
#include <vector_types.h>   // from cuda/include
#include <string>
#include <sstream>

struct CudaImg {
    uint3 size;             // size of picture
    uint channels;          // size of pixel
    union {
        void   *p_void;     // data of picture
        uchar1 *p_uchar1;   // data of picture
        uchar3 *p_uchar3;   // data of picture
        uchar4 *p_uchar4;   // data of picture
    };
    CudaImg(cv::Mat img);  // constructor

    // coordinate getters
    __device__ uchar1& at1(int x, int y) { return p_uchar1[y * size.x + x]; }
    __device__ uchar3& at3(int x, int y) { return p_uchar3[y * size.x + x]; }
    __device__ uchar4& at4(int x, int y) { return p_uchar4[y * size.x + x]; }
};

#endif//__CUDA_IMAGE_H