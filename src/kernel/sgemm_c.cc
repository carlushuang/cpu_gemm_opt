#define _KERNEL_SELECT
#include "sgemm_micro_kernel.h"


/*
*     k            n
* +-------+    +-------+
* |   A   |m   |   B   |k
* +-------+    |       |
*              +-------+
*
*  ---->       n
*           +-----+
*           |  C  | m
*           +-----+
*
* above is actual data in memory, A pannel, B pannel
*
*
* A*B = {a0t, a1t, ..akt} * {b0
*                            b1
                             ...
                             bk}
*
* a0t, a1t,.. -> column vector
* b0, b1, ... -> row vector
*
* A*B = a0t*b0+a1t*b1+...+akt*bk
*
* column vec multiply by row vector -> outer product, can further optimize
* 1) broadcast
* 2) bufferfly/shuffle
*
*
* require A is col major!!!
*/

static void to_tile(int m, int n, int ldc, float * c, float * c_tile){
    int i, j;
    float * ptr_c;
    float * ptr_c_tile = c_tile;
    for(j=0;j<m;j++){
        ptr_c = c+j*ldc;
        for(i=0;i<n;i++){
            *ptr_c_tile++ = *ptr_c++;
        }
    }
}
static void from_tile(int m, int n, int ldc, float * c, float * c_tile){
    int i, j;
    float * ptr_c;
    float * ptr_c_tile = c_tile;
    for(j=0;j<m;j++){
        ptr_c = c+j*ldc;
        for(i=0;i<n;i++){
            *ptr_c++ = *ptr_c_tile++;
        }
    }
}

void sgemm_kernel_c(int m, int n, int k,
    const float * A, const float * B,
    float * C, int ldc)
{
    int x,y,z;
    const float * ptr_a = A;
    const float * ptr_b = B;
    float c_tile[m*n];
    // copy to tile
    to_tile(m,n,ldc,C,c_tile);

    for(z=0; z<k ; z++ ){
        // do outer product of col_vector A and row_vector B
        // broadcast
        /*
        *    b0  b1  b2 ... bn
        * a0 c00 c01        c0n
        * a1 c10 c11        c1n
        * a2 c20 c21        c2n
        * 
        * am cm0 cm1        cmn
        *
        */
       float * c_itr = c_tile;
       ptr_a = A+z*m; // column vector
       for(y=0;y<m;y++){
           ptr_b = B+z*n; // row vector
           for(x=0;x<n;x++){
               *c_itr = *c_itr + (*ptr_a) * (*ptr_b++);
               c_itr++;
           }
           ptr_a++;
       }
    }
    from_tile(m,n,ldc,C,c_tile);
}
