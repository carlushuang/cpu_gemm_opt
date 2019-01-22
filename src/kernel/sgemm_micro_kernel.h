#ifndef __GEMM_KERNEL_H
#define __GEMM_KERNEL_H


void sgemm_micro_kernel(int m, int n, int k,
    float alpha,
    const float  *   A,
    const float *   B,
    float beta,
    float *  C,
    int ldc);


#ifdef _KERNEL_SELECT
#define sgemm_kernel_c sgemm_micro_kernel
//
#endif

#endif
