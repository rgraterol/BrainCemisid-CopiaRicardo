#ifndef COUNTNETWORK_H
#define COUNTNETWORK_H
#define  SIZE_CHARACTERISTIC 32
#define  SIZE_FLAGS          9

struct CountNetwork {
    unsigned char * vectorNetworkCount;
    unsigned char * vectorPointerCount;
    unsigned char * bipPointer;
    unsigned char * clackPointer;

    unsigned char * vectorFlagsCount;
    unsigned char * ptr;
    unsigned char * binaryCharacteristicCount;

};


struct OrderNetwork {

    CountNetwork * countNet;
    unsigned char * vectorNetworkOrder;
    unsigned char * order;
    unsigned char * bumPointer;
    unsigned char * category;
    unsigned char * numRelation;
};

enum stateOrderNetwork{FOUND, NOTFOUND};

#endif // COUNTNETWORK_H
