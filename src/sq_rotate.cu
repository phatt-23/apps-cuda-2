#include "inc/cu_precomp.h"
#include "inc/module.h"

__global__
void cuda_transpone(CudaImg og, CudaImg rot) {
    uint2 pos = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y,
    };
    
    if(pos.x >= og.size.x) return;
    if(pos.y >= og.size.y) return;
    rot.at3(pos.y, pos.x) = og.at3(pos.x, pos.y);  
}

__host__
void cu_sq_rotate(CudaImg og, CudaImg rot, cu_fn::Direction option) {
    // int id = cudaGetDevice(&id);
    
    dim3 bd;
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
    printf("block_dim: (%d, %d, %d)\n", bd.x, bd.y, bd.z);

    dim3 gd(
        (og.size.x + bd.x - 1) / bd.x,
        (og.size.y + bd.y - 1) / bd.y
    );
    printf("grid_dim: (%d, %d, %d)\n", gd.x, gd.y, gd.z);

    // cudaMemPrefetchAsync(og.p_void, og.size.x * og.size.x * sizeof(uchar4), id);

    switch (option) {
        case cu_fn::to_left:
            cuda_transpone<<< gd, bd >>>(og, rot);
            cu_mirror(rot, cu_fn::vertical);
            break;
        case cu_fn::to_right:
            cuda_transpone<<< gd, bd >>>(og, rot);
            cu_mirror(rot, cu_fn::horizontal);
            break;
        default: break;
    };

    cudaDeviceSynchronize();
}
