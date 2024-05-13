#include "inc/module.h"
#include "inc/cu_precomp.h"

__global__
void cuda_mirror(CudaImg og, cu_fn::Mirror option) {
    uint2 pos = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y
    };

    uint tid = pos.y * og.size.x + pos.x;

    // hor
    uint mirror_tid;
    if(option == cu_fn::horizontal)
        mirror_tid = tid + og.size.x - 2 * pos.x - 1;
    else
        mirror_tid = (pos.y + og.size.y - 2 * pos.y) * og.size.x + pos.x - og.size.x;
    
    uchar3 temp = og.p_uchar3[tid];
    og.p_uchar3[tid] = og.p_uchar3[mirror_tid];
    og.p_uchar3[mirror_tid] = temp;
}

__host__
void cu_mirror(CudaImg og, cu_fn::Mirror option) {
    dim3 bd;
    // 1024 = 2^10
    for(size_t i = 10; i > 0; i--) {
        bd.x = 1 << i;
        if((og.size.x / bd.x) % 2 == 0 && (og.size.x % bd.x) == 0) {
            break;
        }
    }

    for(size_t i = 10; i > 0; i--) {
        bd.y = 1 << i;
        if((og.size. y/ bd.y) % 2 == 0 && (og.size.y % bd.y) == 0) {
            break;
        }
    }
    printf(">> block_dim: x= %d, y= %d, z= %d\n", 
        bd.x, bd.y, bd.z);

    // 0 = hor
    // 1 = vert
    
    dim3 gd(
        (og.size.x + bd.x - 1) / bd.x,
        (og.size.y + bd.y - 1) / bd.y
    );
    printf("\tgrid_dim: x= %d, y= %d, z= %d\n", 
        gd.x, gd.y, gd.z);
    
    if (option == cu_fn::horizontal) {
        gd.x /= 2;
    } else {
        gd.y /= 2;
    }
    printf(">> grid_dim: x= %d, y= %d, z= %d\n", 
        gd.x, gd.y, gd.z);

    cuda_mirror<<< gd, bd >>>(og, option);
    
    cudaDeviceSynchronize();
}