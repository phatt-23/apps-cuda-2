#ifndef __CUDA_MODULE_HEADER_FILE_CUH
#define __CUDA_MODULE_HEADER_FILE_CUH

#include "module.h"

__global__
void cuda_kernel_bilinear_resize(CudaImg resized, CudaImg og, float2 scale);

#endif//__CUDA_MODULE_HEADER_FILE_CUH