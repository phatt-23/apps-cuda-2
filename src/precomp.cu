#include "inc/cu_precomp.h"
#include "inc/cu_module.cuh"

void find_optimal2(dim3& gd, dim3& bd, dim3& mat_size) {
    for(size_t i = 10; i > 0; i--) {
        bd.x = 1 << i;
        if((mat_size.x / bd.x) % 2 == 0 && (mat_size.x % bd.x) == 0) {
            break;
        }
    }
    for(size_t i = 10; i > 0; i--) {
        bd.y = 1 << i;
        if((mat_size.y / bd.y) % 2 == 0 && (mat_size.y % bd.y) == 0) {
            break;
        }
    }
    gd.x = ceil((mat_size.x + bd.x) / bd.x);
    gd.y = ceil((mat_size.y + bd.y) / bd.y);
}

void func_gd_bd_info(const char* func_name, dim3& gd, dim3& bd)
{
    if(__DEBUG_FUNC_NAME_GRID_DIM_GRID_DIM_INFO) {
        printf("INFO: '%s' grid_dim: (%d, %d, %d), block_dim: (%d, %d, %d)\n",
            func_name, gd.x, gd.y, gd.z, bd.x, bd.y, bd.z);
    }
}