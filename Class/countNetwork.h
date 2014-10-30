#ifndef COUNTNETWORK_H
#define COUNTNETWORK_H
#define  SIZE_CHARACTERISTIC 32
#define  SIZE_FLAGS          9

struct CountNetwork {
    unsigned char * vectorNetworkCount;
    unsigned char * vectorPointerCount;

    unsigned char * vectorFlagsCount;
    unsigned char * ptr;
    unsigned char * binaryCharacteristicCount;

};


struct OrderNetwork {

    CountNetwork * countNet;
    unsigned char * vectorNetworkOrder;
    unsigned char * category;
};

/*struct SizeCountNet{
    unsigned int numNeuronCount;
    unsigned int sizeVectorNeuronCount;
    unsigned int sizevectorFlagsCount;
    unsigned int sizeBinaryCharacteristicCount;
};*/


#endif // COUNTNETWORK_H