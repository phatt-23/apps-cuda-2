#include "inc/canvas.h"
#include "inc/timing.h"

// MAIN ------------------------------------------------------------------
int main(int argc, char** argv) {
    UniformAllocator allc;
    cv::Mat::setDefaultAllocator(&allc);

    std::vector<cv::Mat> textures = {
        cv::imread("assets/reptile.jpg",   cv::IMREAD_UNCHANGED), // 0
        cv::imread("assets/abstract.jpg",  cv::IMREAD_UNCHANGED), // 1
        cv::imread("assets/cubes.png",     cv::IMREAD_UNCHANGED), // 2
        cv::imread("assets/motorist.jpg",  cv::IMREAD_UNCHANGED), // 3
        cv::imread("assets/ball.png",      cv::IMREAD_UNCHANGED), // 4
        cv::imread("assets/landscape.jpg", cv::IMREAD_UNCHANGED), // 5
    };

    Canvas canvas("New Window :)", textures[5], {600, 400});
    Timing timing(60); // set fps to 60

    CudaRect r_0(textures[4], {20, 20}, {20, 210}, 150); // head
    CudaRect r_1(textures[4], {30, 30}, {40, 205}, 200); // torso
    CudaRect r_2(textures[4], {40, 40}, {70, 200}, 250); // bottom

    int2 direction = { .x = 1, .y = 1 };
    float velocity = 10;

    while(timing.running)
    {
        if(!timing.next()) continue;

        //* update 
        if(r_2.pos.y + r_2.rsize.y >= canvas.size.y) {
            direction.y = -1;
        }
        if(r_2.pos.x + r_2.rsize.x >= canvas.size.x) {
            direction.x = -1;
        }

        if(r_0.pos.y <= 0) {
            direction.y = 1;
        }
        if(r_2.pos.x <= 0) {
            direction.x = 1;
        }

        r_0.add_pos(direction.y * velocity, direction.x * velocity);
        r_1.add_pos(direction.y * velocity, direction.x * velocity);
        r_2.add_pos(direction.y * velocity, direction.x * velocity);

        //* draw
        canvas.flush();
        canvas.draw(r_0);
        canvas.draw(r_1);
        canvas.draw(r_2);

        //* show
        canvas.show();
    }

    return 0;
}












/*
std::vector<CudaImg> init_cuda_images(std::vector<cv::Mat>& cv_images)
{
    std::vector<CudaImg> cuda_images;
    for(size_t i = 0; i < cv_images.size(); ++i) {
        cuda_images.push_back(CudaImg(cv_images[i]));
    }
    return cuda_images;
}

void show_cv_images(std::vector<cv::Mat>& cv_images)
{
    for(size_t i = 0; i < cv_images.size(); ++i) {
        cv::imshow("CV Image " + std::to_string(i), cv_images[i]);
    }
}
*/