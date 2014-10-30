#ifndef LOCK_H
#define LOCK_H
#include "utilCuda.h"

struct Lock {

    int *mutex;

    Lock() {
        int state = 0;
        checkCudaErrors( cudaMalloc(&mutex,sizeof(int) ) );
        checkCudaErrors( cudaMemcpy( mutex, &state, sizeof(int),cudaMemcpyHostToDevice ) );
    }

    ~Lock() {

    }

    __device__ void lock() {
        while( atomicCAS( mutex, 0, 1 ) != 0 );
    }

    __device__ void unlock() {
        atomicExch( mutex, 0 );
    }

    void freeMem(){
        cudaFree(mutex);
    }
};

#endif // LOCK_H
