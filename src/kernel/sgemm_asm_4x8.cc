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
#if 0
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

        a_panel_s4 = _mm256_extractf128_ps(a_panel_s8, 0);    // latency 3
        a_panel_s8_dup = _mm256_set_m128(a_panel_s4, a_panel_s4);    // latency 3
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
    //c0 = _mm256_add_ps(c0, c0_s8);
    //c0 = _mm256_add_ps(c0, c0_s8);
    //c0 = _mm256_add_ps(c0, c0_s8);
    c1 = _mm256_add_ps(c1, c1_s8);
    c2 = _mm256_add_ps(c2, c2_s8);
    c3 = _mm256_add_ps(c3, c3_s8);

    c_ptr = C;
    _mm256_store_ps(c_ptr, c0); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c1); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c2); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c3);
#endif
#if 0
    int k_itr = k/2;
    int k_rem = k%2;
    int k_i;

    __m256 c0_s8, c1_s8, c2_s8, c3_s8;

    const float * ptr_a = A;
    const float * ptr_b = B;

    c0_s8 = _mm256_setzero_ps();
    c1_s8 = _mm256_setzero_ps();
    c2_s8 = _mm256_setzero_ps();
    c3_s8 = _mm256_setzero_ps();

    for(k_i=0; k_i<k_itr ; k_i++ ){
        __m256 a0, a1;
        __m256 b0 = _mm256_load_ps(ptr_b);
        __m256 b1 = _mm256_load_ps(ptr_b+8);

        a0 = _mm256_broadcast_ss(ptr_a);
        a1 = _mm256_broadcast_ss(ptr_a+1);
        c0_s8 = _mm256_fmadd_ps( a0, b0, c0_s8);
        c1_s8 = _mm256_fmadd_ps( a1, b0, c1_s8);
        ptr_a += 2;

        a0 = _mm256_broadcast_ss(ptr_a);
        a1 = _mm256_broadcast_ss(ptr_a+1);
        c2_s8 = _mm256_fmadd_ps( a0, b0, c2_s8);
        c3_s8 = _mm256_fmadd_ps( a1, b0, c3_s8);
        ptr_a += 2;

        a0 = _mm256_broadcast_ss(ptr_a);
        a1 = _mm256_broadcast_ss(ptr_a+1);
        c0_s8 = _mm256_fmadd_ps( a0, b1, c0_s8);
        c1_s8 = _mm256_fmadd_ps( a1, b1, c1_s8);
        ptr_a += 2;

        a0 = _mm256_broadcast_ss(ptr_a);
        a1 = _mm256_broadcast_ss(ptr_a+1);
        c2_s8 = _mm256_fmadd_ps( a0, b1, c2_s8);
        c3_s8 = _mm256_fmadd_ps( a1, b1, c3_s8);
        ptr_a += 2;

        ptr_b += 16;
    }
    for(k_i=0;k_i<k_rem;k_i++){
        __m256 a0;
        __m256 b0 = _mm256_load_ps(ptr_b);

        a0 = _mm256_broadcast_ss(ptr_a);
        ptr_a ++;
        c0_s8 = _mm256_fmadd_ps( a0, b0, c0_s8);

        a0 = _mm256_broadcast_ss(ptr_a);
        ptr_a ++;
        c1_s8 = _mm256_fmadd_ps( a0, b0, c1_s8);

        a0 = _mm256_broadcast_ss(ptr_a);
        ptr_a ++;
        c2_s8 = _mm256_fmadd_ps( a0, b0, c2_s8);

        a0 = _mm256_broadcast_ss(ptr_a);
        ptr_a ++;
        c3_s8 = _mm256_fmadd_ps( a0, b0, c3_s8);

        ptr_b += 8;
    }

    // to global
    __m256 c0, c1, c2, c3;
    float * c_ptr = C;
    c0 = _mm256_load_ps(c_ptr); c_ptr += ldc;
    c1 = _mm256_load_ps(c_ptr); c_ptr += ldc;
    c2 = _mm256_load_ps(c_ptr); c_ptr += ldc;
    c3 = _mm256_load_ps(c_ptr);

    c0 = _mm256_add_ps(c0, c0_s8);
    c1 = _mm256_add_ps(c1, c1_s8);
    c2 = _mm256_add_ps(c2, c2_s8);
    c3 = _mm256_add_ps(c3, c3_s8);

    c_ptr = C;
    _mm256_store_ps(c_ptr, c0); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c1); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c2); c_ptr += ldc;
    _mm256_store_ps(c_ptr, c3);
#endif
#if 1
    // need use 64bit long
    unsigned long long k_itr = k/2;
    unsigned long long k_rem = k%2;
    unsigned long long ldc_ = ldc;

    asm volatile(
    "movq           %2,     %%rax                   \n" // A
    "movq           %3,     %%rbx                   \n" // B
    "                                               \n"
    "vxorps         %%ymm0, %%ymm0, %%ymm0          \n"
    "vxorps         %%ymm1, %%ymm1, %%ymm1          \n"
    "vxorps         %%ymm2, %%ymm2, %%ymm2          \n"
    "vxorps         %%ymm3, %%ymm3, %%ymm3          \n"
    "                                               \n"
    "movq           %0,     %%rsi                   \n" // loop k_itr
    "testq          %%rsi,  %%rsi                   \n"
    "je             .LOOP_ITER_DONE                 \n"
    "                                               \n"
    "vmovaps        (%%rbx),        %%ymm4          \n" // B panel preload
    ".LOOP_ITER:                                    \n"
    "prefetcht0     8*32(%%rax)                     \n"
    "prefetcht0     16*32(%%rbx)                    \n"
    //"vmovaps        (%%rbx),        %%ymm4          \n" // B panel
    "vmovaps        32(%%rbx),      %%ymm5          \n" // B panel + 1
    "                                               \n"
    "vbroadcastss   (%%rax),        %%ymm6          \n"
    "vbroadcastss   4(%%rax),       %%ymm7          \n"
    "vfmadd231ps    %%ymm6, %%ymm4, %%ymm0          \n" // AT&T syntax: vfmadd231ps ymm3, ymm2, ymm1
    "vfmadd231ps    %%ymm7, %%ymm4, %%ymm1          \n" // y1=y2*y3+y1
    "                                               \n" // AT&T syntax: vfmadd213ps ymm3, ymm2, ymm1
    "vbroadcastss   8(%%rax),       %%ymm8          \n" // y1=y2*y1+y3
    "vbroadcastss   12(%%rax),      %%ymm9          \n"
    "vfmadd231ps    %%ymm8, %%ymm4, %%ymm2          \n"
    "vfmadd231ps    %%ymm9, %%ymm4, %%ymm3          \n"
    "vmovaps        (%%rbx),        %%ymm4          \n" // B panel
    "                                               \n"
    "vbroadcastss   16(%%rax),      %%ymm6          \n"
    "vbroadcastss   20(%%rax),      %%ymm7          \n"
    "vfmadd231ps    %%ymm6, %%ymm5, %%ymm0          \n"
    "vfmadd231ps    %%ymm7, %%ymm5, %%ymm1          \n"
    "                                               \n"
    "vbroadcastss   24(%%rax),      %%ymm8          \n"
    "vbroadcastss   28(%%rax),      %%ymm9          \n"
    "vfmadd231ps    %%ymm8, %%ymm5, %%ymm2          \n"
    "vfmadd231ps    %%ymm9, %%ymm5, %%ymm3          \n"
    "                                               \n"
    "addq           $32,    %%rax                   \n"
    "addq           $64,    %%rbx                   \n"
    "                                               \n"
    "subq           $1,     %%rsi                   \n"
    "jne            .LOOP_ITER                      \n"
    "                                               \n"
    ".LOOP_ITER_DONE:                               \n"
    "movq           %1,     %%rsi                   \n" // loop k_rem
    "testq          %%rsi,  %%rsi                   \n"
    "je             .POST                           \n"
    "                                               \n"
    ".LOOP_REM:                                     \n"
    "vmovaps        (%%rbx),        %%ymm4          \n" // B panel
    "vbroadcastss   (%%rax),        %%ymm6          \n"
    "vfmadd231ps    %%ymm4, %%ymm6, %%ymm0          \n"
    "vbroadcastss   4(%%rax),       %%ymm6          \n"
    "vfmadd231ps    %%ymm4, %%ymm6, %%ymm1          \n"
    "vbroadcastss   8(%%rax),       %%ymm6          \n"
    "vfmadd231ps    %%ymm4, %%ymm6, %%ymm2          \n"
    "vbroadcastss   12(%%rax),      %%ymm6          \n"
    "vfmadd231ps    %%ymm4, %%ymm6, %%ymm3          \n"
    "addq           $16,    %%rax                   \n"
    "addq           $32,    %%rbx                   \n"
    "                                               \n"
    "subq           $1,     %%rsi                   \n"
    "jne            .LOOP_REM                       \n"
    "                                               \n"
    ".POST:                                         \n"
    "movq           %4,     %%rax                   \n" // C
    "movq           %5,     %%rdi                   \n"
    "leaq           (%%rax, %%rdi, 4), %%rbx        \n"
    "leaq           (%%rbx, %%rdi, 4), %%rcx        \n"
    "leaq           (%%rcx, %%rdi, 4), %%rdx        \n"
    "vaddps         (%%rax), %%ymm0, %%ymm0         \n" // AT&T syntax: vaddps ymm3, ymm2, ymm1, ymm3+ymm2 -> ymm1
    "vaddps         (%%rbx), %%ymm1, %%ymm1         \n"
    "vaddps         (%%rcx), %%ymm2, %%ymm2         \n"
    "vaddps         (%%rdx), %%ymm3, %%ymm3         \n"
    "vmovaps        %%ymm0, (%%rax)                 \n"
    "vmovaps        %%ymm1, (%%rbx)                 \n"
    "vmovaps        %%ymm2, (%%rcx)                 \n"
    "vmovaps        %%ymm3, (%%rdx)                 \n"
    : // output
    : // input
        "r" (k_itr),  // 0
        "r" (k_rem),  // 1
        "m" (A),      // 2
        "m" (B),      // 3
        "m" (C),      // 4
        "r" (ldc_)    // 5
    : // clobber
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
        "ymm6", "ymm7", "ymm8", "ymm9"
    );
#endif
}