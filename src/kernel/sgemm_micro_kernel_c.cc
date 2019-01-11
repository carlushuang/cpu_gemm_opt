#define _KERNEL_SELECT
#include "sgemm_micro_kernel.h"


/*
*      k           k
* +-------+    +-------+
* |   A   |m   |   B   |n
* +-------+    |       |
*              +-------+
*
*  ---->       n
*           +-----+
*           |  C  | m
*           +-----+
*
* above is actual data in memory, A pannel, B pannel are mutlipled line by line
*
*/
void sgemm_kernel_c(int m, int n, int k,
    float alpha,
    const float * A, const float * B,
    float * C, int ldc)
{
    int x,y,z;

    for(z=0;z<m;z++){
        const float *  b_itr = B;
        float *  c_itr = C;
        for(y=0;y<n;y++){
            const float * a_itr = A;
            float v = 0;

            for(x=0;x<k;x++)
                v += (*a_itr++) * (*b_itr++);

            *c_itr = *c_itr + alpha*v;
            c_itr++;
        }
        C += ldc;
        A += k;
    }
}
