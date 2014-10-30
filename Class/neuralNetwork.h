#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#define  SIZE_CHARACTERISTIC 32
#define  SIZE_FLAGS          9

struct NeuralNetwork {
    unsigned char  * vectorNeuron;
    unsigned char  * vectorFlags;
    unsigned char  * ptr;
    unsigned short * binaryCharacteristic;
};
struct SizeNet{
    unsigned int numNeuron;
    unsigned int sizeVectorNeuron;
    unsigned int sizevectorFlags;
    unsigned int sizeBinaryCharacteristic;
};

enum senses{SIGHT, HEARING, SMELL, TOUCH, TASTE};

enum stateNeuralNetwork{IS_HIT,NO_HIT,DIFF};

//     0   1   2   3   4  5   6   7   8
enum{ KNW,HIT,DGR,DIS,MBR,CAT,CON,RAT,RED};

#endif // NEURALNETWORK_H
