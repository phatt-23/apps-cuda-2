#ifndef __CUDA_MODULE_HEADER_FILE_CUH
#define __CUDA_MODULE_HEADER_FILE_CUH

#include "module.h"

#define __DEBUG_FUNC_NAME_GRID_DIM_GRID_DIM_INFO 0

void func_gd_bd_info(const char* func_name, dim3& gd, dim3& bd);

#endif//__CUDA_MODULE_HEADER_FILE_CUH