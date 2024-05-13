#include "inc/timing.h"
#include <opencv4/opencv2/opencv.hpp>

Timing::Timing(int t_fps) 
    : fps(t_fps) 
    , frame_duration(1.0 / t_fps * (float)1E6)
    , running(1)
{
    std::cout << "INFO: Timing created!" << std::endl;
	gettimeofday(&this->old, NULL);
	this->start = this->old;
}

Timing::Timing()
    : fps(60)
    , frame_duration(1.0 / 60.0 * (float)1E6)
    , running(1)
{
	gettimeofday(&this->old, NULL);
	this->start = this->old;
}

Timing::~Timing()
{
    std::cout << "INFO: Timing destroyed!" << std::endl;
}


int Timing::next()
{
    // key 'ESC' is pressed
    int key = cv::waitKey(1);
    if(key == 27) {
        std::cout << "INFO: ESC key pressed!" << std::endl;
        this->running = 0;
        return 0;
    }

    gettimeofday(&this->current, NULL);
    timersub(&this->current, &this->old, &this->delta);
	
    if(this->delta.tv_usec < this->frame_duration) return 0; // too short time
	
    this->iterations++;
    this->old = this->current;
	this->delta_sec = (float) this->delta.tv_usec / 1E6;

    return 1;
}
