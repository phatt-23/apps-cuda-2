#ifndef __TIMING_H
#define __TIMING_H

#include <stdio.h>
#include <sys/time.h>
#include <iostream>

class Timing
{
    public:
        int frame_duration;  
        int fps;
        float delta = 0.f; // time in seconds
        size_t iterations = 0; // number of iterations
        timeval _start, _current, _old, _delta;
        bool running;

        Timing();
        Timing(int fps);
        ~Timing();
        int next();
};

#endif//__TIMING_H

