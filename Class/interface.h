#ifndef INTERFACE_H
#define INTERFACE_H
#include <iostream>

struct Interface
{

    unsigned char *arrayCategory;
    unsigned char *id;
    int *hits;

    Interface();
    void setHit();
    void freeMem(bool deleteAll = false);
};

#endif // INTERFACE_H
