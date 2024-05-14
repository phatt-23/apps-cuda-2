#include "inc/module.h"
#include "inc/canvas.h"
#include "inc/timing.h"


int main(int argc, char** argv) {
    __DEBUG_INFO_CU_INSERT_IMAGE = 0; // toggle debug console printout
    __DEBUG_INFO_KERNEL_LAUNCH   = 0;
    UniformAllocator allc;
    cv::Mat::setDefaultAllocator(&allc);

    std::vector<cv::Mat> textures = { // load in textures
        cv::imread("assets/reptile.jpg",   cv::IMREAD_UNCHANGED), // 0
        cv::imread("assets/abstract.jpg",  cv::IMREAD_UNCHANGED), // 1
        cv::imread("assets/cubes.png",     cv::IMREAD_UNCHANGED), // 2
        cv::imread("assets/motorist.jpg",  cv::IMREAD_UNCHANGED), // 3
        cv::imread("assets/ball.png",      cv::IMREAD_UNCHANGED), // 4
        cv::imread("assets/landscape.jpg", cv::IMREAD_UNCHANGED), // 5
        cv::Mat::zeros({800,700}, CV_8UC4), // 6
        cv::Mat::zeros({800,700}, CV_8UC3), // 7
    };

    Canvas canvas("Window", textures[0], {600,1200});
    CudaRect rect_ball(textures[4], {canvas.size.y - 40, 500}, {40,20}, 180);
    Timing timing(60);

    int direction = -1;

    while(timing.running)
    {
        if(!timing.next()) continue;
        canvas.flush();
        //* update
        if(rect_ball.pos.y < 400) 
            direction = 1;
        else if(rect_ball.pos.y + rect_ball.rsize.y > canvas.size.y) 
            direction = -1;

        rect_ball.add_pos(direction * 300 * timing.delta, 0);

        //* draw
        canvas.draw(rect_ball);

        //* show
        canvas.show();
    }

    return 0;
}






























#if 0
// MAIN ------------------------------------------------------------------
int main(int argc, char** argv) {
    UniformAllocator allc;
    cv::Mat::setDefaultAllocator(&allc);

    std::vector<cv::Mat> textures = { // load in textures
        cv::imread("assets/reptile.jpg",   cv::IMREAD_UNCHANGED), // 0
        cv::imread("assets/abstract.jpg",  cv::IMREAD_UNCHANGED), // 1
        cv::imread("assets/cubes.png",     cv::IMREAD_UNCHANGED), // 2
        cv::imread("assets/motorist.jpg",  cv::IMREAD_UNCHANGED), // 3
        cv::imread("assets/ball.png",      cv::IMREAD_UNCHANGED), // 4
        cv::imread("assets/landscape.jpg", cv::IMREAD_UNCHANGED), // 5
    };
    // cv::Mat cv_image = cv::Mat::zeros({800, 800}, CV_8UC3);
    // CudaImg cuda_motorist(textures[3]);
    // CudaImg cuda_image(cv_image);
    // cu_bilinear_resize(cuda_image, cuda_motorist);
    // cv::imshow("window", cv_image);
    // cv::waitKey(0);

    Canvas canvas("New Window :)", textures[5], {600, 400});
    Timing timing(60); // set fps to 60

    CudaRect rect2(textures[2], {50, 50}, {200, 200});
    CudaRect rect(textures[4], {300, 300}, {40, 40}, 255); // ball

    int2 direction = { .x = 1, .y = 1 };
    float velocity = 10;

    while(timing.running)
    {
        if(!timing.next()) continue;

        //* update 
        if(rect.pos.y + rect.rsize.y >= canvas.size.y) {
            direction.y = -1;
        }
        if(rect.pos.x + rect.rsize.x >= canvas.size.x) {
            direction.x = -1;
        }

        if(rect.pos.y <= 0) {
            direction.y = 1;
        }
        if(rect.pos.x <= 0) {
            direction.x = 1;
        }

        rect.add_pos(direction.y * velocity, direction.x * velocity);

        //* draw
        canvas.flush();
        canvas.draw(rect2);
        canvas.draw(rect);

        //* show
        canvas.show();
    }

    return 0;
}
#endif