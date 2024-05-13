#include "inc/cu_precomp.h"
#include "inc/module.h"

__global__
void cuda_merge_half(CudaImg og, CudaImg im, uint option) {
    uint2 p = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y,
    };

    if(p.x >= og.size.x || p.y >= og.size.y) return;

    switch (option) {
        case cu_fn::left_right: 
            og.at3(p.x, p.y) = im.at3(p.x + og.size.x/2 - 1, p.y);
            break;
        case cu_fn::right_left: 
            og.at3(p.x + og.size.x/2 - 1, p.y) = im.at3(p.x, p.y);
            break;
        case cu_fn::top_bottom:
            og.at3(p.x, p.y) = im.at3(p.x, p.y + og.size.y/2 - 1);
            break;
        case cu_fn::bottom_top:
            og.at3(p.x, p.y + og.size.y/2 - 1) = im.at3(p.x, p.y);
            break;
        default: 
            og.at3(p.x, p.y) = im.at3(p.x, p.y);
            break;
    };
}

void cu_merge_half(CudaImg og, CudaImg im, cu_fn::MergeHalf option) {
    if(option > 7) {
        printf("ERROR: %s, the option can only be of [0..7], you gave %d\n", __PRETTY_FUNCTION__, option);
        return;
    }

    if(og.size.x != im.size.x || og.size.y != im.size.y) {
        printf("ERROR: %s, the images are of differing sizes\n", __PRETTY_FUNCTION__);
        return;
    }

    dim3 bd, gd;
    dim3 mat_size(og.size.x, og.size.y);
    find_optimal2(gd, bd, mat_size);
    printf("INFO: grid_dim: (%d, %d, %d), block_dim: (%d, %d, %d)\n",
        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z
    );

    if(option <= 3)    
        gd.x >>= 1;
    else if(option <= 7)
        gd.y >>= 1;

    printf(">> INFO: grid_dim: (%d, %d, %d), block_dim: (%d, %d, %d)\n",
        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z
    );

    cuda_merge_half<<<gd, bd>>>(og, im, option);

    cudaDeviceSynchronize();
}
