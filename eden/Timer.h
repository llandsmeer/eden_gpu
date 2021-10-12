//
// Created by max on 04-10-21.
//

#ifndef EDEN_GPU_TIMER_H
#define EDEN_GPU_TIMER_H

struct Timer {
    timeval start;
    Timer() {
        gettimeofday(&start, 0);
    }
    double delta() {
        timeval end;
        gettimeofday(&end, 0);
        return TimevalDeltaSec(start, end);
    }
};

#endif