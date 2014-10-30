#include "interface.h"

Interface::Interface()
{
    arrayCategory = NULL;
    id  = NULL;
    hits= new int;
}
void Interface::setHit()
{
    arrayCategory=new unsigned char [*hits];
    id= new unsigned char [*hits];
}

void Interface::freeMem(bool deleteAll)
{
    if(arrayCategory != NULL)
        delete arrayCategory;

    if(id != NULL)
        delete id;

    if(deleteAll)
        delete hits;
}
