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
