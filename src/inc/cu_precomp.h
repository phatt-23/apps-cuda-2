#ifndef __CUDA_PRECOMP_H
#define __CUDA_PRECOMP_H

#include "uni_mem.h"
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <cstddef>

extern bool __DEBUG_INFO_KERNEL_LAUNCH;
extern bool __DEBUG_INFO_CU_INSERT_IMAGE;

// finds optimal size for the kernel launch 
void find_optimal2(dim3& gd, dim3& bd, dim3& mat_size);

// prints out the grid dim size and block dim size
void func_gd_bd_info(const char* func_name, dim3& gd, dim3& bd);

// check for cuda errors
void check_cuda_error(const char *func_name, int line);

#endif//__CUDA_PRECOMP_H