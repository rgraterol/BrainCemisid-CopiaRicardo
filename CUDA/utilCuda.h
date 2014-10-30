#ifndef UTILCUDA_H
#define UTILCUDA_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <string>
#include <stdio.h>
using std::string;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        string cad;
        char buffer [33];
        sprintf(buffer,"%d",line);
        cad.append("CUDA error at: ");
        cad.append(file);
        cad.append(":");
        cad.append(buffer);
        cad.append("\n");
        cad.append(cudaGetErrorString(err));
        cad.append(" " );
        cad.append(func);
        cad.append("\n");
        throw cad;
    }
}

#endif // UTILCUDA_H
