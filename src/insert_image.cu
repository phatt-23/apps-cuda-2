#include "inc/module.h"

inline static void check_within_bounds(const char *func_name, const uint3 &big_size, const uint3 &small_size, const uint2 &pos)
{
    if (__DEBUG_INFO_CU_INSERT_IMAGE) {
        if (big_size.x - small_size.x > pos.x >= 0 &&
            big_size.y - small_size.y > pos.y >= 0)
        {
            printf("NOTE: %s => Insering image (%d x %d) into (%d x %d) at (%d, %d).\n",
                   func_name, small_size.x, small_size.y, big_size.x, big_size.y, pos.x, pos.y);
        }
        else 
        {
            printf("WARN: %s => Insering image (%d x %d) into (%d x %d) at (%d, %d) will overfill.\n",
                   func_name, small_size.x, small_size.y, big_size.x, big_size.y, pos.x, pos.y);
        }
    }
}
//
//
//
__global__ 
void cuda_kernel_insert_rgb_image(CudaImg big, CudaImg small, uint2 pos, float alpha)
{
    uint2 s = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y,
    };
    uint2 b = {.x = pos.x + s.x,
               .y = pos.y + s.y};

    if (s.x >= small.size.x || s.y >= small.size.y)
        return;
    if (b.x >= big.size.x || b.y >= big.size.y)
        return;

    uchar3 bp = big.at3(b.x, b.y);
    uchar3 sp = small.at3(s.x, s.y);
    uchar3 np;

    np.x = bp.x * (UINT8_MAX - alpha) / UINT8_MAX + sp.x * (alpha / UINT8_MAX);
    np.y = bp.y * (UINT8_MAX - alpha) / UINT8_MAX + sp.y * (alpha / UINT8_MAX);
    np.z = bp.z * (UINT8_MAX - alpha) / UINT8_MAX + sp.z * (alpha / UINT8_MAX);

    big.at3(b.x, b.y) = np;
}

__host__
void cu_insert_rgb_image(CudaImg& big, CudaImg& small, uint2 pos, uint8_t alpha)
{
    // check if the images are of 3 channels
    if (big.channels != 3 || small.channels != 3)
    {
        printf("ERROR: %s => Only works with RGB images.\n", __PRETTY_FUNCTION__);
        return;
    }
    check_within_bounds(__PRETTY_FUNCTION__, big.size, small.size, pos);
    dim3 gd, bd, mat_size(small.size.x, small.size.y);
    find_optimal2(gd, bd, mat_size);

    func_gd_bd_info("cu_insert_rgb_image", gd, bd);

    cuda_kernel_insert_rgb_image<<<gd, bd>>>(big, small, pos, alpha);

    check_cuda_error(__PRETTY_FUNCTION__, __LINE__);
    cudaDeviceSynchronize();
}
//
//
//
__global__ 
void cuda_kernel_insert_rgba_image(CudaImg big, CudaImg small, uint2 pos, float alpha)
{
    uint2 s = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y,
    };
    uint2 b = {.x = pos.x + s.x,
               .y = pos.y + s.y};

    if (s.x >= small.size.x || s.y >= small.size.y)
        return;
    if (b.x >= big.size.x || b.y >= big.size.y)
        return;

    uchar3 bp = big.at3(b.x, b.y);
    uchar4 sp = small.at4(s.x, s.y);
    int3 np;

    alpha = sp.w * alpha / 255;

    np.x = bp.x * (255 - alpha) / 255 + sp.x * alpha / 255;
    np.y = bp.y * (255 - alpha) / 255 + sp.y * alpha / 255;
    np.z = bp.z * (255 - alpha) / 255 + sp.z * alpha / 255;

    np.x = bp.x * (UINT8_MAX - alpha) / UINT8_MAX + sp.x * (alpha / UINT8_MAX);
    np.y = bp.y * (UINT8_MAX - alpha) / UINT8_MAX + sp.y * (alpha / UINT8_MAX);
    np.z = bp.z * (UINT8_MAX - alpha) / UINT8_MAX + sp.z * (alpha / UINT8_MAX);

    big.at3(b.x, b.y).x = (uchar)np.x;
    big.at3(b.x, b.y).y = (uchar)np.y;
    big.at3(b.x, b.y).z = (uchar)np.z;
}
//
//
__global__ 
void cuda_kernel_insert_rgba_image_in_rgba_image(CudaImg big, CudaImg small, uint2 pos, float alpha)
{
    uint2 s = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y,
    };
    uint2 b = {.x = pos.x + s.x,
               .y = pos.y + s.y};

    if (s.x >= small.size.x || s.y >= small.size.y) return;
    if (b.x >= big.size.x || b.y >= big.size.y) return;

    big.at4(b.x, b.y) = small.at4(s.x, s.y);
}

__global__ 
void cuda_kernel_insert_rgb_image_in_rgba_image(CudaImg big, CudaImg small, uint2 pos, float alpha)
{
    uint2 s = {
        .x = blockDim.x * blockIdx.x + threadIdx.x,
        .y = blockDim.y * blockIdx.y + threadIdx.y,
    };
    uint2 b = {.x = pos.x + s.x,
               .y = pos.y + s.y};

    if (s.x >= small.size.x || s.y >= small.size.y) return;
    if (b.x >= big.size.x || b.y >= big.size.y) return;

    uchar3 np = small.at3(s.x, s.y);
    big.at4(b.x, b.y) = make_uchar4(np.x, np.y, np.z, uchar(alpha/(float)255));
}

typedef void (*kernel_func)(CudaImg big, CudaImg small, uint2 pos, float alpha);
static kernel_func insert_image_kernel_funcs[4] = {
    &cuda_kernel_insert_rgb_image,                  // 00
    &cuda_kernel_insert_rgb_image_in_rgba_image,    // 01
    &cuda_kernel_insert_rgba_image,                 // 10
    &cuda_kernel_insert_rgba_image_in_rgba_image,   // 11
};

__host__
void cu_insert_rgba_image(CudaImg& big, CudaImg& small, uint2 pos, uint8_t alpha)
{
    check_within_bounds(__PRETTY_FUNCTION__, big.size, small.size, pos);

    dim3 gd, bd, mat_size(small.size.x, small.size.y);
    find_optimal2(gd, bd, mat_size);
    func_gd_bd_info("cu_insert_rgba_image", gd, bd);

    uint index = 0;
    if(big.channels == 4)   index |= 0b01;
    if(small.channels == 4) index |= 0b10;

    insert_image_kernel_funcs[index]<<<gd, bd>>>(big, small, pos, alpha);

    check_cuda_error(__PRETTY_FUNCTION__, __LINE__);
    cudaDeviceSynchronize();
}
