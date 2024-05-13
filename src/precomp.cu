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