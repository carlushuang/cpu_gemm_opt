#include "util.h"

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
double current_sec()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void* __aligned_malloc(size_t required_bytes, size_t alignment)
{
    if( alignment==0 || (alignment & (alignment-1)))    // check pow of 2
        return NULL;
    void* p1; // original block
    void** p2; // aligned block
    int offset = alignment - 1 + sizeof(void*);
    if ((p1 = (void*)malloc(required_bytes + offset)) == NULL)
    {
       return NULL;
    }
    p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void __aligned_free(void *p)
{
    free(((void**)p)[-1]);
}

void rand_vector_f32(float * v, int elem) {
    int i;

    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }

    for(i=0;i<elem;i++){
        v[i] = ((float)(rand() % 100)) / 100.0f;
    }
}