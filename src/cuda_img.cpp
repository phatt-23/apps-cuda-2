#include "inc/cuda_img.h"

CudaImg::CudaImg(cv::Mat& img)
    : channels(img.channels())
    , p_void((void*) img.data)
    , p_cv_mat(&img)
{
    this->size.x = img.size().width;
    this->size.y = img.size().height;
}

