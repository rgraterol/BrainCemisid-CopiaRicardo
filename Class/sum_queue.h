#ifndef SUM_QUEUE_H
#define SUM_QUEUE_H

#include <iostream>


class SumQueue
{
    private:
        struct node
        {
            int num;
            struct node *next;
        };

        struct queue
        {
            node *foward;
            node *back  ;
        };
     public:
       void enqueue(struct queue &q, int value);
       int dequeue(struct queue &q);
       void showQueue( struct queue q );
       void clearQueue( struct queue &q);

};
#endif // SUM_QUEUE_H
