#define _KERNEL_SELECT
#include "sgemm_micro_kernel.h"



static void tile_to_global(int m, int n, int ldc,
    float * c, float * c_tile,
    float alpha, float beta)
{
    int i, j;
    float * ptr_c;
    float * ptr_c_tile = c_tile;
    for(j=0;j<m;j++){
        ptr_c = c+j*ldc;
        for(i=0;i<n;i++){
            *ptr_c = *ptr_c_tile * alpha + *ptr_c*beta;
            ptr_c++;
            ptr_c_tile++;
        }
    }
}
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
* A pannel col major, B pannel row major
*
*
* A*B = {a0t, a1t, ..akt} * {b0
*                            b1
*                            ...
*                            bk}
*
* a0t, a1t,.. -> column vector
* b0, b1, ... -> row vector
*
* A*B = a0t*b0 + a1t*b1 +...+ akt*bk
*
* column vec multiply by row vector -> outer product, can further optimize
* 1) broadcast
* 2) bufferfly/shuffle
*
* require A is col major!!!
*/

void sgemm_kernel_c(int m, int n, int k,
    float alpha,
    const float * A, const float * B,
    float beta,
    float * C, int ldc)
{
    int x,y,z;
    const float * ptr_a = A;
    const float * ptr_b = B;
    float c_tile[m*n];
    int i;
    for(i=0;i<m*n;i++)
        c_tile[i] = 0;

    //to_tile(m,n,ldc,C,c_tile, beta);

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
    tile_to_global(m,n,ldc,C,c_tile,1,1);
}
