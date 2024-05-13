#include "inc/cu_precomp.h"
#include "inc/cuda_img.h"

__global__
void cuda_split_bgr(CudaImg og, CudaImg b, CudaImg g, CudaImg r) {
    uint2 pos = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y,
    };
    
    if(pos.x >= og.size.x) return;
    if(pos.y >= og.size.y) return;

    uint tid = pos.y * og.size.x + pos.x;

    uchar3 color = og.p_uchar3[tid];

    b.p_uchar3[tid].x = color.x;
    g.p_uchar3[tid].y = color.y;
    r.p_uchar3[tid].z = color.z;
}

__host__
void cu_split_bgr(CudaImg og, CudaImg b, CudaImg g, CudaImg r) {
    dim3 block_dim(
        16, 16, 1
    );
    dim3 grid_dim(
        (og.size.x + block_dim.x - 1) / block_dim.x,
        (og.size.y + block_dim.y - 1) / block_dim.y,
        1
    );

    cuda_split_bgr<<<grid_dim, block_dim>>>(og, b, g, r);

    cudaDeviceSynchronize();
}


