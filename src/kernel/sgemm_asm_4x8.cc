#define _KERNEL_SELECT
#include "sgemm_micro_kernel.h"

//#include <x86intrin.h>

#include <immintrin.h> // AVX2
#include <assert.h>

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
*/
void sgemm_asm_4x8(int m, int n, int k,
    float alpha,
    const float * A, const float * B,
    float beta,
    float * C, int ldc)
{
    assert(m==4 && n==8 && "4x8 kernel");
    int z;
    __m256 c0_0_8, c1_0_8, c2_0_8, c3_0_8;

    const float * ptr_a = A;
    const float * ptr_b = B;

    c0_0_8 = _mm256_setzero_ps();
    c1_0_8 = _mm256_setzero_ps();
    c2_0_8 = _mm256_setzero_ps();
    c3_0_8 = _mm256_setzero_ps();

    for(z=0; z<k ; z++ ){
        __m256 a_tmp;
        __m256 b_0 = _mm256_load_ps(ptr_b);
        a_tmp = _mm256_broadcast_ss(ptr_a);
        c0_0_8 = _mm256_fmadd_ps( a_tmp, b_0, c0_0_8);
        ptr_a++;

        a_tmp = _mm256_broadcast_ss(ptr_a);
        c1_0_8 = _mm256_fmadd_ps( a_tmp, b_0, c1_0_8);
        ptr_a++;

        a_tmp = _mm256_broadcast_ss(ptr_a);
        c2_0_8 = _mm256_fmadd_ps( a_tmp, b_0, c2_0_8);
        ptr_a++;

        a_tmp = _mm256_broadcast_ss(ptr_a);
        c3_0_8 = _mm256_fmadd_ps( a_tmp, b_0, c3_0_8);
        ptr_a++;

        ptr_b += n;
    }

    // to global
    __m256 c_tmp;
    c_tmp = _mm256_load_ps(C);
    c_tmp = _mm256_add_ps(c_tmp, c0_0_8);
    _mm256_store_ps(C, c_tmp);
    C += ldc;

    c_tmp = _mm256_load_ps(C);
    c_tmp = _mm256_add_ps(c_tmp, c1_0_8);
    _mm256_store_ps(C, c_tmp);
    C += ldc;

    c_tmp = _mm256_load_ps(C);
    c_tmp = _mm256_add_ps(c_tmp, c2_0_8);
    _mm256_store_ps(C, c_tmp);
    C += ldc;

    c_tmp = _mm256_load_ps(C);
    c_tmp = _mm256_add_ps(c_tmp, c3_0_8);
    _mm256_store_ps(C, c_tmp);
    //C += ldc;
}