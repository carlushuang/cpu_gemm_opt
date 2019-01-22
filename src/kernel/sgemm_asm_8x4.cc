#define _KERNEL_SELECT
#include "sgemm_micro_kernel.h"

//#include <x86intrin.h>
#include <immintrin.h> // AVX
#include <immintrin.h> // AVX2


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
* MR=8, NR=4, k=iter, prefer multiple of 8
*/
void sgemm_asm_8x4(int m, int n,int k,
    float alpha,
    const float  *   A,
    const float *   B,
    float *  C,
    int ldc)
{
    
#if 0
    int k_itr = k/8;
    int k_rem = k%8;
    asm volatile(
        "movq           %0, %%rax\n"
        "movq           %1, %%rbx\n"
        "movq           %2, %%rcx\n"
        "movq           %3, %%rsi\n"
        "movq           %4, %%rdi\n"
        :  // output
            
        :  // input
          "m"(A),       // 0
          "m"(B),       // 1
          "m"(C),       // 2
          "r"(k_itr),   // 3
          "r"(k_rem)    // 4
        :// clobber
    );
#endif
}
