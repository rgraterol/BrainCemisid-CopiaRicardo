#ifndef STATISTIC_H
#define STATISTIC_H
#include <float.h>
struct Statistic{

    int numExecutions;
    float maxTime;
    float minTime;
    float accumulateTime;

    Statistic(){
        numExecutions    = 0;
        maxTime          = FLT_MIN;
        minTime          = FLT_MAX;
        accumulateTime   = 0.0;
    }
};

enum kernels{BOOT,RECOGNIZE,CORRECT,RESET };

#endif // STATISTIC_H
