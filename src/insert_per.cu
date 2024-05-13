#include "inc/cu_precomp.h"
#include "inc/module.h"

__global__
void cuda_insert_per(CudaImg og, CudaImg im, uint k, cu_fn::Position s) {
    uint2 p = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y
    };
    if(p.x >= og.size.x || p.y >= og.size.y) return;

    switch (s) {
        case cu_fn::pos_top:
            og.at3(p.x, p.y) = im.at3(p.x, p.y);
            break;
        case cu_fn::pos_bottom:
            og.at3(p.x, ((100.f - k)/100.0f * og.size.y) + p.y) = 
                im.at3(p.x, ((100.f - k)/100.f * og.size.y) + p.y);
            break;
        case cu_fn::pos_left:
            og.at3(((100.f - k)/100.0f * og.size.x) + p.x, p.y) = 
                im.at3(((100.f - k)/100.0f * og.size.x) + p.x, p.y);
            break;
        case cu_fn::pos_right:
            og.at3(p.x, p.y) = im.at3(p.x, p.y);
            break;
    }
}

__host__
void cu_insert_per(CudaImg& og, CudaImg& im, uint p, cu_fn::Position x) {
    if(p > 100) {
        printf("ERROR: %s => Cannot put the percentage bigger than 100, you gave %d\n",
            __PRETTY_FUNCTION__, p);
        return;
    }
    dim3 m_size( og.size.x, og.size.y );
    dim3 gd, bd;
    find_optimal2(gd, bd, m_size);

    printf("INFO: insert_per => grid_dim: (%d, %d, %d), block_dim: (%d, %d, %d)\n",
        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z
    );
    switch (x) {
        case cu_fn::pos_top:
            gd.y = ceil(gd.y * ((float)p/100.f)); 
            break;
        case cu_fn::pos_bottom:
            gd.y = ceil(gd.y * ((float)p/100.f)); 
            break;
        case cu_fn::pos_left:
            gd.x = ceil(gd.x * ((float)p/100.f)); 
            break;
        case cu_fn::pos_right:
            gd.x = ceil(gd.x * ((float)p/100.f)); 
            break;
        default: break;
    };
    printf(">>> INFO: insert_per => grid_dim: (%d, %d, %d), block_dim: (%d, %d, %d)\n",
        gd.x, gd.y, gd.z, bd.x, bd.y, bd.z
    );

    cuda_insert_per<<<gd, bd>>>(og, im, p, x);

    cudaDeviceSynchronize();
}
