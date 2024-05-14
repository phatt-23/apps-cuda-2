#include "inc/module.h"
#include "inc/cu_module.cuh"

__device__
uchar3 bilinear_point3(CudaImg og, float2 o, float2 d)
{
    uchar3 p00 = og.at3(    (int)o.x,     (int)o.y); // top left
    uchar3 p01 = og.at3(    (int)o.x, 1 + (int)o.y); // bottom left
    uchar3 p10 = og.at3(1 + (int)o.x,     (int)o.y); // top right
    uchar3 p11 = og.at3(1 + (int)o.x, 1 + (int)o.y); // bottom right

    uchar3 p = { // pixel to be put inside resized image
        .x = (uchar)
             (p00.x * (1 - d.x) * (1 - d.y) +
              p10.x * (d.x)     * (1 - d.y) +
              p01.x * (1 - d.x) * (d.y) +
              p11.x * (d.x)     * (d.y)), 
        .y = (uchar)
             (p00.y * (1 - d.x) * (1 - d.y) +
              p10.y * (d.x)     * (1 - d.y) +
              p01.y * (1 - d.x) * (d.y) +
              p11.y * (d.x)     * (d.y)),
        .z = (uchar)
             (p00.z * (1 - d.x) * (1 - d.y) +
              p10.z * (d.x)     * (1 - d.y) +
              p01.z * (1 - d.x) * (d.y) +
              p11.z * (d.x)     * (d.y)),
    };

    return p;
}

__device__
uchar4 bilinear_point4(CudaImg og, float2 o, float2 d)
{
    uchar4 p00 = og.at4(    (int)o.x,     (int)o.y); // top left
    uchar4 p01 = og.at4(    (int)o.x, 1 + (int)o.y); // bottom left
    uchar4 p10 = og.at4(1 + (int)o.x,     (int)o.y); // top right
    uchar4 p11 = og.at4(1 + (int)o.x, 1 + (int)o.y); // bottom right

    uchar4 p = { // pixel to be put inside resized image
        .x = (uchar)
             (p00.x * (1 - d.x) * (1 - d.y) +
              p10.x * (d.x)     * (1 - d.y) +
              p01.x * (1 - d.x) * (d.y) +
              p11.x * (d.x)     * (d.y)), 
        .y = (uchar)
             (p00.y * (1 - d.x) * (1 - d.y) +
              p10.y * (d.x)     * (1 - d.y) +
              p01.y * (1 - d.x) * (d.y) +
              p11.y * (d.x)     * (d.y)),
        .z = (uchar)
             (p00.z * (1 - d.x) * (1 - d.y) +
              p10.z * (d.x)     * (1 - d.y) +
              p01.z * (1 - d.x) * (d.y) +
              p11.z * (d.x)     * (d.y)),
        .w = (uchar)
             (p00.w * (1 - d.x) * (1 - d.y) +
              p10.w * (d.x)     * (1 - d.y) +
              p01.w * (1 - d.x) * (d.y) +
              p11.w * (d.x)     * (d.y)),
    };

    return p;
}

__global__
void cuda_kernel_bilinear_resize3(CudaImg resized, CudaImg og, float2 scale)
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
    
    float2 d = { // difference from the 'original' coordinate
        .x = o.x - (int)o.x,
        .y = o.y - (int)o.y,
    };

    resized.at3(r.x, r.y) = bilinear_point3(og, o, d); // putting pixel inside
}

__global__
void cuda_kernel_bilinear_resize4(CudaImg resized, CudaImg og, float2 scale)
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
    
    float2 d = { // difference from the 'original' coordinate
        .x = o.x - (int)o.x,
        .y = o.y - (int)o.y,
    };

    resized.at4(r.x, r.y) = bilinear_point4(og, o, d); // putting pixel inside
}

void cu_bilinear_resize(CudaImg resized, CudaImg og)
{
    dim3 gd, bd, mat_size(resized.size.x, resized.size.y);
    find_optimal2(gd, bd, mat_size);
    func_gd_bd_info("cu_bilinear_resize", gd, bd);

    float2 scale = { // scale in 'x' and 'y' axes
        .x = (og.size.x - 1) / float(resized.size.x),
        .y = (og.size.y - 1) / float(resized.size.y),
    };
    
    if(resized.channels == 3)
        cuda_kernel_bilinear_resize3<<<gd, bd>>>(resized, og, scale);
    else if(resized.channels == 4)   
        cuda_kernel_bilinear_resize4<<<gd, bd>>>(resized, og, scale);

    cudaError_t cu_err;
    if((cu_err = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA ERROR: %s, %d => %s\n", __FILE__, __LINE__, cudaGetErrorString(cu_err));
    }
    
    cudaDeviceSynchronize();
}

