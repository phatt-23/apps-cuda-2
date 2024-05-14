#include "inc/timing.h"
#include <opencv4/opencv2/opencv.hpp>

Timing::Timing(int t_fps) 
    : fps(t_fps) 
    , frame_duration(1.0 / t_fps * (float)1E6)
    , running(1)
{
    std::cout << "INFO: Timing created! (FPS = " << this->fps << ")" << std::endl;
	gettimeofday(&this->_old, NULL);
	this->_start = this->_old;
}

Timing::Timing()
    : fps(60)
    , frame_duration(1.0 / 60.0 * (float)1E6)
    , running(1)
{
    std::cout << "INFO: Timing created! (FPS = " << this->fps << ")" << std::endl;
	gettimeofday(&this->_old, NULL);
	this->_start = this->_old;
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

    gettimeofday(&this->_current, NULL);
    timersub(&this->_current, &this->_old, &this->_delta);
    
    // skip this iteration (too short time)
    if(this->_delta.tv_usec < this->frame_duration) return 0;
	
    this->iterations++;
    this->_old = this->_current;
	this->delta = (float)this->_delta.tv_usec / 1E6;

    return 1;
}
