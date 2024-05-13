#ifndef __MODULE_H
#define __MODULE_H

#include "cu_precomp.h"
#include "cuda_img.h"

namespace cu_fn {
    enum MergeHalf {
        left_right,
        left_left,
        right_left,
        right_right,
        top_bottom,
        top_top,
        bottom_top,
        bottom_bottom,
    };
    enum Mirror {
        horizontal,
        vertical,
    };
    enum Direction {
        to_left,
        to_right,
    };
    enum Position {
        pos_top,
        pos_right,
        pos_bottom,
        pos_left,
    };
    enum Axis {
        axis_x,
        axis_y,
        axis_xy,
    };
};

/// Breaks up the R, G, B elements from an image.
void cu_split_bgr(CudaImg og, CudaImg b, CudaImg g, CudaImg r);

/// Mirrors a given image, horizontaly `0` or vertically `1`.
void cu_mirror(CudaImg og, cu_fn::Mirror option);

/// Rotates an image into another image. 
void cu_sq_rotate(CudaImg og, CudaImg rotated, cu_fn::Direction option);

/// Embeds an half of the `im` image into the `og`.
void cu_merge_half(CudaImg og, CudaImg im, cu_fn::MergeHalf option);

/// Embeds a precentage `p` of the `im` image 
/// into `og` provided the starting position `x`.
void cu_insert_per(CudaImg& og, CudaImg& im, uint p, cu_fn::Position x);

/// Scales the image 2x in different axes.
cv::Mat cu_scale2x_img(CudaImg og, cu_fn::Axis a);

/// Inserts a small image (src) into a bigger image (dest).
/// Works only with RGB images (no alpha channel).
void cu_insert_rgb_image(CudaImg big, CudaImg small, uint2 pos, uint8_t alpha);

/// Inserts a small image into a bigger image.
/// Works with RGBA and RGB images.
void cu_insert_rgba_image(CudaImg big, CudaImg small, uint2 pos, uint8_t alpha);

/// Resizes the og image to fit the resize image.
/// Will write to the resized.
/// Using bilinear interpolation.
void cu_nearest_neighbour_resize(CudaImg resized, CudaImg og);

/// Resizes the og image to fit the resize image.
/// Will write to the resized.
/// Using bilinear interpolation.
void cu_bilinear_resize(CudaImg resized, CudaImg og);

/// NOT IMPLEMENTED
/// Resizes the src image to fit the dest image.
/// Will write to the resized.
/// Using bicubic interpolation.
// void cu_bicubic_resize(CudaImg dest, CudaImg src);

#endif//__MODULE_H