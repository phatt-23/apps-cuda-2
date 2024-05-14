#include "inc/cu_precomp.h"
#include "inc/module.h"
#include "inc/cu_module.cuh"

__global__
void cuda_scale2x_img(CudaImg og, CudaImg sc, cu_fn::Axis a) {
    uint2 p = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y,
    };

    if(p.x >= og.size.x || p.y >= og.size.y) return;


    // sc.at4( 2*p.x,       2*p.y       ) = og.at4(p.x, p.y);
    // sc.at4( 2*(p.x + 1), 2*p.y       ) = og.at4(p.x, p.y);
    // sc.at4( 2*p.x,       2*(p.y + 1) ) = og.at4(p.x, p.y);
    // sc.at4( 2*(p.x + 1), 2*(p.y + 1) ) = og.at4(p.x, p.y);
    switch (a) {
        case cu_fn::axis_x:
            sc.at3( p.x + p.x + 1, p.y ) = og.at3( p.x, p.y );
            sc.at3( p.x + p.x, p.y ) = og.at3( p.x, p.y );
            break;
        case cu_fn::axis_y:
            sc.at3( p.x + p.x + 1, p.y ) = og.at3( p.x, p.y );
            sc.at3( p.x + p.x, p.y ) = og.at3( p.x, p.y );
            break;
        default: break;
    }

}

__host__
cv::Mat cu_scale2x_img(CudaImg og, cu_fn::Axis a) {
    dim3 gd;
    dim3 bd;
    dim3 mat_size(og.size.x, og.size.y);
    find_optimal2(gd, bd, mat_size);


    cv::Mat cv_scaled (
        (cv::Size) { 
            int(2 * og.size.x), 
            int(2 * og.size.y), 
        }, 
        CV_8UC3
    );
    CudaImg scaled(cv_scaled);
    
    printf("INFO: Mat Original => cv::Size = (%d, %d)\n", 
        og.size.y, og.size.x);
    printf("INFO: Mat 2x       => cv::Size = (%d, %d)\n", 
        scaled.size.y, scaled.size.x);
        
    func_gd_bd_info("cu_scale2x_img", gd, bd);

    cuda_scale2x_img<<<gd, bd>>>(og, scaled, a);

    cudaDeviceSynchronize();
    return cv_scaled;
}