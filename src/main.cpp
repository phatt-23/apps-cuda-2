#include "inc/module.h"
#include "inc/canvas.h"
#include "inc/timing.h"

inline int randi32(int bottom, int upper)
{
    return rand() % (upper - bottom) + bottom;
}

int main(int argc, char** argv) {
    __DEBUG_INFO_CU_INSERT_IMAGE = 1; // toggle debug console printout
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
        cv::Mat::zeros({150,450}, CV_8UC4), // 6 - snowman
        cv::Mat::zeros({800,700}, CV_8UC3), // 7
    };

    CudaImg img_bg(textures[5]);
    CudaImg img_ball(textures[4]);
    CudaImg img_snowman(textures[6]);

    Canvas snowman("snowman", textures[6], {textures[6].rows, textures[6].cols}, CV_8UC4);

    CudaRect bottom(textures[4],{snowman.size.y - 150, 0}, {150, 150});
    CudaRect torso(textures[4],
        { snowman.size.y - bottom.rsize.y - 120, 
          ((snowman.size.x - 120) / 2) }, 
        {120, 120}
    );
    CudaRect head(textures[4],
        { snowman.size.y - bottom.rsize.y - torso.rsize.y - 90, 
          ((snowman.size.x - 90) / 2) }, 
        {90, 90}
    );

    snowman.draw(bottom);
    snowman.draw(torso);
    snowman.draw(head);


    // cu_insert_rgba_image(img_bg, snowman.canvas, {0}, 255);

    // skok
    Canvas anim("aanim", *img_bg.p_cv_mat, {720,1080});

    Timing t(60);
    int direction = -1;
    CudaRect rect_snowman(*snowman.canvas.p_cv_mat, {anim.size.y - 120, 300}, {120, 90});
    
    while(t.running) {
        if (!t.next()) continue;

        if(rect_snowman.pos.y < 300)
            direction = 1;
        else if(rect_snowman.pos.y + rect_snowman.rsize.y > anim.size.y) {
            direction = -1;
            break;
        }
        
        rect_snowman.add_pos(direction * 200 * t.delta, 0);

        anim.flush();
        anim.draw(rect_snowman);
        anim.show();
    }

    for (int i = 0; i < 50; i++)
    {
        cv::waitKey(50);
        rect_snowman.set_pos(
            randi32(0, anim.size.y - rect_snowman.rsize.y),
            randi32(0, anim.size.x - rect_snowman.rsize.x)
        ); 
        anim.draw(rect_snowman);
        anim.show();
    }

    cv::waitKey(0);
    return 0;
}


#if 0
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
#endif






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