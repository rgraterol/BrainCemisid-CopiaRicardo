#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utilCuda.h"
#include "timer.h"
#include "lock.h"
#include "../Class/interface.h"
#include "../Class/neuralNetwork.h"
#include "../Class/statistic.h"


#define  RATIO               22
#define  MIN_RATIO           2

//cuda kernel prototypes
__global__ void  correct  ( unsigned char * d_vectorFlags , unsigned char * d_ptr, unsigned char *d_desiredOutput);
__global__ void  reset    ( unsigned char * d_vectorFlags , unsigned char *d_ptr);
__global__ void  recognize( unsigned char * d_vectorNeuron, unsigned char *d_vectorFlags,
                            unsigned char * d_pattern     , int *d_countHit, unsigned char *d_ptr,
                            unsigned char * d_arrayCategory  , unsigned char* d_idsNeuron,Lock lock);
// methods prototype
template<class T>
inline bool equal(T a, T b);

template<class T>
bool compare(T array [] , int sizeArray);

void calculateStatistic(const float & currentTime, Statistic * & statistic, kernels kernel);
void debugTimer( GpuTimer timer);

//--------------------------------------Metodos Main-------------------------------
extern "C"
void boot(NeuralNetwork * & neuralSenses,const SizeNet & sizeNet, Statistic * & statistic){
    unsigned char * d_vectorZero;
    GpuTimer timer;

    // It allocates memory on the device
    checkCudaErrors(cudaMalloc(&d_vectorZero,sizeNet.sizeVectorNeuron));

    // initialize the memory block to zero (0)
    timer.Start();
    checkCudaErrors(cudaMemset(d_vectorZero , 0 , sizeNet.sizeVectorNeuron));
    timer.Stop();

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    calculateStatistic(timer.Elapsed(),statistic,BOOT);

    // copy from device to host
    checkCudaErrors(cudaMemcpy(neuralSenses[ SIGHT ].vectorNeuron, d_vectorZero, sizeNet.sizeVectorNeuron, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(neuralSenses[ SIGHT ].vectorFlags , d_vectorZero, sizeNet.sizevectorFlags , cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(neuralSenses[ SIGHT ].binaryCharacteristic,d_vectorZero,sizeNet.sizeBinaryCharacteristic,cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(neuralSenses[ HEARING ].vectorNeuron, d_vectorZero, sizeNet.sizeVectorNeuron, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(neuralSenses[ HEARING ].vectorFlags , d_vectorZero, sizeNet.sizevectorFlags , cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(neuralSenses[ HEARING ].binaryCharacteristic,d_vectorZero,sizeNet.sizeBinaryCharacteristic,cudaMemcpyDeviceToHost));

    // Free memory on device Reserved
    checkCudaErrors(cudaFree(d_vectorZero));
}

extern "C"
stateNeuralNetwork recognize(NeuralNetwork * neuralSenses, const SizeNet & sizeNet,
                             unsigned char * h_pattern, Interface * interface, Statistic *& statistic)
{
    int * d_countHit;
    unsigned char * d_arrayCategory,*d_idsNeuron;
    unsigned char * d_vectorNeuron,* d_vectorFlags,*d_pattern,*d_ptr;
    stateNeuralNetwork state;

    dim3 blockSize (SIZE_CHARACTERISTIC);
    dim3 gridSize  ( (*neuralSenses->ptr) +1 );
    GpuTimer timer;
    Lock lock;

    *(interface->hits) = 0;

    // It allocates memory on the device
    checkCudaErrors(cudaMalloc( &d_vectorNeuron, sizeNet.sizeVectorNeuron) );
    checkCudaErrors(cudaMalloc( &d_vectorFlags , sizeNet.sizevectorFlags ) );
    checkCudaErrors(cudaMalloc( &d_pattern     , sizeof(unsigned char) * SIZE_CHARACTERISTIC));
    checkCudaErrors(cudaMalloc( &d_arrayCategory  , sizeof(unsigned char) * (*(neuralSenses->ptr))));
    checkCudaErrors(cudaMalloc( &d_idsNeuron   , sizeof(unsigned char) * (*(neuralSenses->ptr))));
    checkCudaErrors(cudaMalloc( &d_ptr         , sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc( &d_countHit    , sizeof(int)));

    // copy from host to device
    checkCudaErrors( cudaMemcpy( d_vectorNeuron, neuralSenses->vectorNeuron ,sizeNet.sizeVectorNeuron, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_vectorFlags , neuralSenses->vectorFlags  ,sizeNet.sizevectorFlags , cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_pattern     , h_pattern                  ,sizeof(unsigned char)*SIZE_CHARACTERISTIC, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_ptr         , neuralSenses->ptr          ,sizeof(unsigned char)   , cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_countHit    , interface->hits            ,sizeof(int)             , cudaMemcpyHostToDevice ) );

    //call kernel reconize
    timer.Start();
    recognize<<<gridSize,blockSize>>>(d_vectorNeuron,d_vectorFlags,d_pattern,d_countHit,d_ptr,d_arrayCategory,d_idsNeuron,lock);
    timer.Stop();

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    calculateStatistic(timer.Elapsed(),statistic,RECOGNIZE);

    // copy from device to host
    checkCudaErrors( cudaMemcpy( interface->hits, d_countHit, sizeof(int), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy( neuralSenses->vectorNeuron , d_vectorNeuron, sizeNet.sizeVectorNeuron, cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy( neuralSenses->vectorFlags  , d_vectorFlags , sizeNet.sizevectorFlags , cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy( neuralSenses->ptr          , d_ptr         , sizeof(unsigned char)   , cudaMemcpyDeviceToHost ) );
    interface->freeMem();
    interface->setHit();
    checkCudaErrors(cudaMemcpy(interface->arrayCategory,d_arrayCategory ,sizeof(unsigned char)*(* (interface->hits)),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(interface->id        ,d_idsNeuron  ,sizeof(unsigned char)*(* (interface->hits)),cudaMemcpyDeviceToHost));

    if(* (interface->hits) > 1){
        if(* (interface->hits) == 2)
            state = equal(interface->arrayCategory[0],interface->arrayCategory[1])? IS_HIT : DIFF;
        else
            state = compare(interface->arrayCategory,* (interface->hits)) ? IS_HIT : DIFF;
    }
    else if(* (interface->hits) == 1)
        state=IS_HIT;
    else
        state=NO_HIT;

    // Free memory on device Reserved
    checkCudaErrors(cudaFree(d_vectorNeuron));
    checkCudaErrors(cudaFree(d_vectorFlags));
    checkCudaErrors(cudaFree(d_pattern));
    checkCudaErrors(cudaFree(d_countHit));
    checkCudaErrors(cudaFree(d_ptr));
    checkCudaErrors(cudaFree(d_arrayCategory));
    checkCudaErrors(cudaFree(d_idsNeuron));
    lock.freeMem();

    return state;
}

extern "C"
void correct(NeuralNetwork * neuralSenses , const SizeNet & sizeNet,
             unsigned char   desiredOutput, int maxThreadsPerBlock, Statistic *&statistic){

    unsigned char * d_desiredOutput;
    unsigned char * d_vectorFlags,* d_ptr;

    dim3 blockSize (maxThreadsPerBlock);
    int numblock= (*(neuralSenses->ptr) % maxThreadsPerBlock == 0) ?
                *(neuralSenses->ptr) / maxThreadsPerBlock:
                *(neuralSenses->ptr) / maxThreadsPerBlock + 1;
    dim3 gridSize(numblock);
    GpuTimer timer;

    // It allocates memory on the device
    checkCudaErrors(cudaMalloc(&d_vectorFlags  ,sizeof(unsigned char) * SIZE_FLAGS* (*neuralSenses->ptr)));
    checkCudaErrors(cudaMalloc(&d_desiredOutput,sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_ptr,sizeof(unsigned char)));

    // copy from host to device
    checkCudaErrors( cudaMemcpy( d_vectorFlags  , neuralSenses->vectorFlags  ,sizeof(unsigned char) * SIZE_FLAGS * (*neuralSenses->ptr), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_ptr          , neuralSenses->ptr          ,sizeof(unsigned char)   , cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_desiredOutput, &desiredOutput             ,sizeof(unsigned char)   , cudaMemcpyHostToDevice ) );

    timer.Start();
    //call kernel reconize
    correct<<<gridSize,blockSize>>>(d_vectorFlags,d_ptr,d_desiredOutput);
    timer.Stop();
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    calculateStatistic(timer.Elapsed(),statistic,CORRECT);

    // copy from device to host
    checkCudaErrors( cudaMemcpy( neuralSenses->vectorFlags , d_vectorFlags , sizeof(unsigned char) * SIZE_FLAGS *(*neuralSenses->ptr), cudaMemcpyDeviceToHost ) );

    // Free memory on device Reserved

    checkCudaErrors(cudaFree(d_vectorFlags));
    checkCudaErrors(cudaFree(d_ptr));
    checkCudaErrors(cudaFree(d_desiredOutput));
}

extern "C"
void reset(NeuralNetwork * neuralSenses , const SizeNet & sizeNet, int maxThreadsPerBlock, Statistic *&statistic)
{   
    unsigned char * d_vectorFlags,* d_ptr;

    dim3 blockSize (maxThreadsPerBlock);
    int numblock = (*(neuralSenses->ptr) % maxThreadsPerBlock == 0) ?
                *(neuralSenses->ptr) / maxThreadsPerBlock:
                *(neuralSenses->ptr) / maxThreadsPerBlock + 1;
    dim3 gridSize(numblock);
    GpuTimer timer;

    // It allocates memory on the device
    checkCudaErrors(cudaMalloc(&d_vectorFlags  ,sizeof(unsigned char) * SIZE_FLAGS * (*neuralSenses->ptr)));
    checkCudaErrors(cudaMalloc(&d_ptr,sizeof(unsigned char)));

    // copy from host to device
    checkCudaErrors( cudaMemcpy( d_vectorFlags  , neuralSenses->vectorFlags  ,sizeof(unsigned char) * SIZE_FLAGS * (*neuralSenses->ptr), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_ptr          , neuralSenses->ptr          ,sizeof(unsigned char)   , cudaMemcpyHostToDevice ) );

    timer.Start();
    //call kernel reconize
    reset<<<gridSize,blockSize>>>(d_vectorFlags,d_ptr);
    timer.Stop();
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    calculateStatistic(timer.Elapsed(),statistic,RESET);

    // copy from device to host
    checkCudaErrors( cudaMemcpy( neuralSenses->vectorFlags , d_vectorFlags , sizeof(unsigned char) * SIZE_FLAGS * (*neuralSenses->ptr), cudaMemcpyDeviceToHost ) );

    // Free memory on device Reserved

    checkCudaErrors(cudaFree(d_vectorFlags));
    checkCudaErrors(cudaFree(d_ptr));
}

// methods
template<class T>
bool equal(T a, T b){
    return (a==b)?  true : false;
}

template<class T>
bool compare(T array[], int sizeArray)
{
    T element=array[0];
    for (register int i = 1; i < sizeArray; i++) {

        if(!equal(element,array[i]))
            return false;
    }

    return true;
}

//cuda kernel
__global__ void  recognize(unsigned char * d_vectorNeuron, unsigned char *d_vectorFlags,
                           unsigned char *d_pattern, int *d_countHit, unsigned char *d_ptr,
                           unsigned char *d_arrayCategory, unsigned char *d_idsNeuron, Lock lock){

    __shared__ unsigned char sharedVectorNeuron     [SIZE_CHARACTERISTIC];
    __shared__ unsigned char sharedVectorFlags      [SIZE_FLAGS];
    __shared__ unsigned char sharedPattern          [SIZE_CHARACTERISTIC];
    __shared__ int           sharedDistanceManhattan[SIZE_CHARACTERISTIC];

    int vectorIndex = threadIdx.x + SIZE_CHARACTERISTIC * blockIdx.x;
    int flagIndex   = threadIdx.x + SIZE_FLAGS  * blockIdx.x;
    int threadIndex = threadIdx.x;
    unsigned char ptr= *d_ptr;

    sharedVectorNeuron[threadIndex] = d_vectorNeuron [vectorIndex];
    sharedPattern     [threadIndex] = d_pattern      [threadIndex];

    if(threadIndex < SIZE_FLAGS)
        sharedVectorFlags[threadIndex]=d_vectorFlags[flagIndex];

    __syncthreads();            // make sure entire block is loaded!

    if(blockIdx.x == ptr)       //si estoy en la neurona lista para aprender copio el patron
    {
        d_vectorNeuron[vectorIndex]           = sharedPattern[threadIndex];

        if(threadIndex == 0)
            d_vectorFlags [ptr * SIZE_FLAGS + RAT]= RATIO;
    }

    else if(sharedVectorFlags[KNW] == 1 && sharedVectorFlags[DGR]==0)
    {
        sharedDistanceManhattan[threadIndex]= fabsf(sharedPattern[threadIndex]-sharedVectorNeuron[threadIndex]);
        __syncthreads();

        // do reduction in shared mem
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIndex < s)
                sharedDistanceManhattan[threadIndex]+= sharedDistanceManhattan[threadIndex+s];

            __syncthreads();        // make sure all adds at one stage are done!
        }

        // only thread 0 writes result for this block back to global mem
        if (threadIndex == 0)
        {
            if(sharedDistanceManhattan[0] < sharedVectorFlags[RAT])
            {
                d_vectorFlags[SIZE_FLAGS  * blockIdx.x + DIS] = sharedDistanceManhattan[0];
                d_vectorFlags[SIZE_FLAGS  * blockIdx.x + HIT] = 1;

                lock.lock();

                d_arrayCategory [*d_countHit] = sharedVectorFlags[CAT];
                d_idsNeuron  [*d_countHit] = blockIdx.x;
                (*d_countHit)++;

                lock.unlock();

            }
        }
    }
}

__global__ void correct(unsigned char *d_vectorFlags, unsigned char *d_ptr, unsigned char *d_desiredOutput)
{
    int indexGlobal=threadIdx.x + blockDim.x * blockIdx.x;

    unsigned char ratio,hit,dis,category;

    if(indexGlobal < *d_ptr)
    {
        hit   = d_vectorFlags[indexGlobal * SIZE_FLAGS + HIT];
        dis   = d_vectorFlags[indexGlobal * SIZE_FLAGS + DIS];
        category   = d_vectorFlags[indexGlobal * SIZE_FLAGS + CAT];
        ratio = d_vectorFlags[indexGlobal * SIZE_FLAGS + RAT];

        if(hit==1 && category != *d_desiredOutput)
        {
            if(ratio > dis) //NECESARIO?
                d_vectorFlags[ indexGlobal * SIZE_FLAGS + RAT ] = dis;

            if(ratio < MIN_RATIO)
                d_vectorFlags[ indexGlobal * SIZE_FLAGS + DGR ] = 1;
        }
    }
}

__global__ void reset(unsigned char *d_vectorFlags, unsigned char *d_ptr)
{
    int indexGlobal=threadIdx.x + blockDim.x * blockIdx.x;

    if(indexGlobal < *d_ptr)
        d_vectorFlags[ indexGlobal * SIZE_FLAGS + HIT ] = 0;
}


void calculateStatistic(const float &currentTime, Statistic *&statistic, kernels kernel)
{
     statistic[kernel].numExecutions++;
     statistic[kernel].accumulateTime += currentTime;

     if(statistic[kernel].minTime >currentTime)
         statistic[kernel].minTime = currentTime;

     if(statistic [kernel].maxTime < currentTime)
         statistic[kernel].maxTime =currentTime;
}

void debugTimer(GpuTimer timer){

    int err = printf("\n%f msecs.\n", timer.Elapsed());

    if (err < 0) {
        //Couldn't print! Probably closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }
}
