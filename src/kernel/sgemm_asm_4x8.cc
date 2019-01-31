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
#if 1
    int k_itr = k/2;
    int k_rem = k%2;
    int k_i;

    const float * ptr_a = A;
    const float * ptr_b = B;

    __m256  c0_s8, c1_s8, c2_s8, c3_s8;
    
    c0_s8 = _mm256_setzero_ps();
    c1_s8 = _mm256_setzero_ps();
    c2_s8 = _mm256_setzero_ps();
    c3_s8 = _mm256_setzero_ps();

    for(k_i=0;k_i<k_itr;k_i++){
        __m256 a_panel_s8;
        __m256 a_panel_s8_dup;
        __m256 a_panel_s8_dup_0;
        __m256 a_panel_s8_dup_1;
        __m256 a_panel_s8_dup_2;
        __m256 a_panel_s8_dup_3;
        __m128 a_panel_s4;

        __m256 b_panel_s8_0;
        __m256 b_panel_s8_1;

        // load all we want in this loop
        a_panel_s8 = _mm256_load_ps(ptr_a);
        b_panel_s8_0 = _mm256_load_ps(ptr_b);
        b_panel_s8_1 = _mm256_load_ps(ptr_b+8);

        a_panel_s4 = _mm256_extractf128_ps(a_panel_s8, 0);
        a_panel_s8_dup = _mm256_set_m128(a_panel_s4, a_panel_s4);
        a_panel_s8_dup_0 = _mm256_permute_ps(a_panel_s8_dup, 0x00);
        a_panel_s8_dup_1 = _mm256_permute_ps(a_panel_s8_dup, 0x55);
        a_panel_s8_dup_2 = _mm256_permute_ps(a_panel_s8_dup, 0xaa);
        a_panel_s8_dup_3 = _mm256_permute_ps(a_panel_s8_dup, 0xff);

        c0_s8 = _mm256_fmadd_ps(a_panel_s8_dup_0, b_panel_s8_0, c0_s8);
        c1_s8 = _mm256_fmadd_ps(a_panel_s8_dup_1, b_panel_s8_0, c1_s8);
        c2_s8 = _mm256_fmadd_ps(a_panel_s8_dup_2, b_panel_s8_0, c2_s8);
        c3_s8 = _mm256_fmadd_ps(a_panel_s8_dup_3, b_panel_s8_0, c3_s8);

        a_panel_s4 = _mm256_extractf128_ps(a_panel_s8, 1);
        a_panel_s8_dup = _mm256_set_m128(a_panel_s4, a_panel_s4);
        a_panel_s8_dup_0 = _mm256_permute_ps(a_panel_s8_dup, 0x00);
        a_panel_s8_dup_1 = _mm256_permute_ps(a_panel_s8_dup, 0x55);
        a_panel_s8_dup_2 = _mm256_permute_ps(a_panel_s8_dup, 0xaa);
        a_panel_s8_dup_3 = _mm256_permute_ps(a_panel_s8_dup, 0xff);

        c0_s8 = _mm256_fmadd_ps(a_panel_s8_dup_0, b_panel_s8_1, c0_s8);
        c1_s8 = _mm256_fmadd_ps(a_panel_s8_dup_1, b_panel_s8_1, c1_s8);
        c2_s8 = _mm256_fmadd_ps(a_panel_s8_dup_2, b_panel_s8_1, c2_s8);
        c3_s8 = _mm256_fmadd_ps(a_panel_s8_dup_3, b_panel_s8_1, c3_s8);

        ptr_a += 8;
        ptr_b += 16;
    }
    for(k_i=0;k_i<k_rem;k_i++){
        __m256 a_panel_s8;
        __m256 a_panel_s8_dup;
        __m128 a_panel_s4;
        __m256 b_panel_s8;

        a_panel_s4 = _mm_load_ps(ptr_a);
        b_panel_s8 = _mm256_load_ps(ptr_b);

        a_panel_s8 = _mm256_set_m128(a_panel_s4, a_panel_s4);

        a_panel_s8_dup = _mm256_permute_ps(a_panel_s8, 0x00);
        c0_s8 = _mm256_fmadd_ps(a_panel_s8_dup, b_panel_s8, c0_s8);

        a_panel_s8_dup = _mm256_permute_ps(a_panel_s8, 0x55);
        c1_s8 = _mm256_fmadd_ps(a_panel_s8_dup, b_panel_s8, c1_s8);

        a_panel_s8_dup = _mm256_permute_ps(a_panel_s8, 0xaa);
        c2_s8 = _mm256_fmadd_ps(a_panel_s8_dup, b_panel_s8, c2_s8);

        a_panel_s8_dup = _mm256_permute_ps(a_panel_s8, 0xff);
        c3_s8 = _mm256_fmadd_ps(a_panel_s8_dup, b_panel_s8, c3_s8);

        ptr_a += 4;
        ptr_b += 8;
    }

    // store to global
    __m256 c0, c1, c2, c3;
    float * c_ptr = C;
    c0 = _mm256_load_ps(c_ptr); c_ptr += ldc;
    c1 = _mm256_load_ps(c_ptr); c_ptr += ldc;
    c2 = _mm256_load_ps(c_ptr); c_ptr += ldc;
    c3 = _mm256_load_ps(c_ptr);
    c0 = _mm256_add_ps(c0, c0_s8);
    c0 = _mm256_add_ps(c0, c0_s8);
    c0 = _mm256_add_ps(c0, c0_s8);
    c0 = _mm256_add_ps(c0, c0_s8);

    c_ptr = C;
    _mm256_store_ps(c_ptr, c0); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c1); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c2); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c3);
#endif
#if 0
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
#endif
}