#include "inc/cuda_rect.h"

CudaRect::CudaRect(cv::Mat& texture, int2 pos)
    : CudaImg(texture)
    , cv_mat_ptr(&texture)
    , pos({(float)pos.y, (float)pos.x})
    , rsize({0, 0})
    , alpha(255)
{
    std::cout << "INFO: CudaRect created! (ogsize = {" 
        << this->size.y << "," << this->size.x << "}, pos = {" 
        << this->pos.y << "," << this->pos.x <<"})" << std::endl;
}

CudaRect::CudaRect(cv::Mat& texture, int2 pos, uint2 rsize)
    : CudaImg(texture)
    , cv_mat_ptr(&texture)
    , pos({(float)pos.y, (float)pos.x})
    , rsize({rsize.y, rsize.x})
    , alpha(255)
{
    std::cout << "INFO: CudaRect created! (rsize = {" 
        << this->rsize.y << "," << this->rsize.x << "}, ogsize = {" 
        << this->size.y << "," << this->size.x << "}, pos = {" 
        << this->pos.y << "," << this->pos.x <<"})" << std::endl;
}

CudaRect::CudaRect(cv::Mat& texture, int2 pos, uint alpha)
    : CudaImg(texture)
    , cv_mat_ptr(&texture)
    , rsize({0, 0})
    , pos({(float)pos.y, (float)pos.x})
    , alpha(alpha)
{
    std::cout << "INFO: CudaRect created! (ogsize = {" 
        << this->size.y << "," << this->size.x << "}, pos = {" 
        << this->pos.y << "," << this->pos.x <<"}, alpha = " 
        << this->alpha << ")" << std::endl;
}

CudaRect::CudaRect(cv::Mat& texture, int2 pos, uint2 rsize, uint alpha)
    : CudaImg(texture)
    , cv_mat_ptr(&texture)
    , rsize({rsize.y, rsize.x})
    , pos({(float)pos.y, (float)pos.x})
    , alpha(alpha)
{
    std::cout << "INFO: CudaRect created! (rsize = {" 
        << this->rsize.y << "," << this->rsize.x << "}, ogsize = {" 
        << this->size.y << "," << this->size.x << "}, pos = {" 
        << this->pos.y << "," << this->pos.x <<"}, alpha = " 
        << this->alpha << ")" << std::endl;
}

CudaRect::~CudaRect()
{
    std::cout << "INFO: CudaRect destroyd!" << std::endl;
}

void CudaRect::set_pos(float y, float x)
{
    this->pos.x = x;
    this->pos.y = y;
}

void CudaRect::add_pos(float y, float x) // adds these values to the pos
{
    this->pos.x += x;
    this->pos.y += y;
}
