#include "inc/cuda_rect.h"

CudaRect::CudaRect(cv::Mat& texture, int2 pos, uint alpha)
    : CudaImg(texture), alpha(alpha), rsize({0, 0}), cv_mat_ptr(&texture)
{
    this->pos.x = (float)pos.y;
    this->pos.y = (float)pos.x;
    std::cout << "INFO: CudaRect created! (ogsize = {" << this->size.y << "," << this->size.x << "})" << std::endl;
}

CudaRect::CudaRect(cv::Mat& texture, uint2 rsize, int2 pos, uint alpha)
    : CudaImg(texture), alpha(alpha), cv_mat_ptr(&texture)
{
    this->pos.x = (float)pos.y;
    this->pos.y = (float)pos.x;
    this->rsize.x = rsize.y;
    this->rsize.y = rsize.x;
    std::cout << "INFO: CudaRect created! (rsize = {" << this->rsize.y << "," << this->rsize.x << "}, ogsize = {" << this->size.y << "," << this->size.x << "})" << std::endl;
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
