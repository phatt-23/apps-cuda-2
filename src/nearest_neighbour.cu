#include "inc/module.h"

__global__
void cuda_kernel_nearest_neighbour_resize(CudaImg resized, CudaImg og, float2 scale) 
{
    float2 r = { // resized image coordinate
        .x = float(blockIdx.x * blockDim.x + threadIdx.x),
        .y = float(blockIdx.y * blockDim.y + threadIdx.y),
    };
    if(r.x >= resized.size.x || r.y >= resized.size.y) return;

    float2 o = { // original image coordinate relative to 'r'
        .x = r.x * scale.x,
        .y = r.y * scale.y,
    };
    if(o.x >= og.size.x || o.y >= og.size.y) return;

    resized.at3(r.x, r.y) = og.at3(o.x, o.y);
}

void cu_nearest_neighbour_resize(CudaImg& dest, CudaImg& src)
{
    dim3 gd, bd, mat_size(dest.size.x, dest.size.y);
    find_optimal2(gd, bd, mat_size);
    printf("INFO: grid_dim: (%d, %d, %d), block_dim: (%d, %d, %d)\n",
        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z
    );
    float2 scale = { // scale in 'x' and 'y' axes
        .x = (src.size.x - 1) / float(dest.size.x),
        .y = (src.size.y - 1) / float(dest.size.y),
    };
    cuda_kernel_nearest_neighbour_resize<<<gd, bd>>>(dest, src, scale);

    check_cuda_error(__PRETTY_FUNCTION__, __LINE__);    
    cudaDeviceSynchronize(); 
}
