#ifndef __CUDA_PRECOMP
#define __CUDA_PRECOMP

#include "uni_mem.h"
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <cstddef>

void find_optimal2(dim3& gd, dim3& bd, dim3& mat_size);

#endif//__CUDA_PRECOMP