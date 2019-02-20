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
void sgemm_asm_8x8(int m, int n, int k,
    float alpha,
    const float * A, const float * B,
    float beta,
    float * C, int ldc)
{
#if 1
    assert(m==8 && n==8 && "8x8 kernel");
    unsigned long long k_itr = k/2;
    unsigned long long k_rem = k%2;
    unsigned long long ldc_ = ldc;

    // efficiency, 50%. permute method is not so good :(, they reuse port 5 pressure
    asm volatile(
        "movq           %2,         %%rax               \n" // A
        "movq           %3,         %%rbx               \n" // B
        "                                               \n"
        "vxorps         %%ymm8,  %%ymm8,  %%ymm8        \n"
        "vxorps         %%ymm9,  %%ymm9,  %%ymm9        \n"
        "vxorps         %%ymm10, %%ymm10, %%ymm10       \n"
        "vxorps         %%ymm11, %%ymm11, %%ymm11       \n"
        "vxorps         %%ymm12, %%ymm12, %%ymm12       \n"
        "vxorps         %%ymm13, %%ymm13, %%ymm13       \n"
        "vxorps         %%ymm14, %%ymm14, %%ymm14       \n"
        "vxorps         %%ymm15, %%ymm15, %%ymm15       \n"
        "                                               \n"
        "movq           %0,         %%rsi               \n" // k_itr
        "testq          %%rsi,      %%rsi               \n"
        "je             .LOOP_ITER_DONE                 \n"
        "                                               \n"
        "subq           $1,         %%rsi               \n"
        "vmovaps        (%%rax), %%ymm0                 \n" // preload A
        "vmovaps        (%%rbx), %%ymm1                 \n" // preload B
        "testq          %%rsi,      %%rsi               \n"
        "je             .LOOP_ITER_LEFT                 \n"
        "                                               \n"
        ".LOOP_ITER:                                    \n"
        "vperm2f128     $0x1, %%ymm1,%%ymm1,%%ymm7      \n" // pre permute cross lane
        "vmovaps        32(%%rax), %%ymm2               \n" // lsb       ...       msb
        "vmovaps        32(%%rbx), %%ymm3               \n" 
        "                                               \n" // b0,b1,b2,b3,b4,b5,b6,b7
        "vpermilps      $0xB1,   %%ymm1, %%ymm4         \n" // b1,b0,b3,b2,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm1, %%ymm5         \n" // b2,b3,b0,b1,b6,b7,b4,b5
        "vpermilps      $0x1B,   %%ymm1, %%ymm6         \n" // b3,b2,b1,b0,b7,b6,b5,b4
        "vfmadd231ps    %%ymm1,  %%ymm0, %%ymm8         \n"
        "vfmadd231ps    %%ymm4,  %%ymm0, %%ymm9         \n"
        "vfmadd231ps    %%ymm5,  %%ymm0, %%ymm10        \n"
        "vfmadd231ps    %%ymm6,  %%ymm0, %%ymm11        \n" // INDEED, port is serialized, not so efficient
        "                                               \n"
        //"vperm2f128     $0x1, %%ymm1,%%ymm1,%%ymm1      \n"
        "                                               \n" // b4,b5,b6,b7,b0,b1,b2,b3
        "vpermilps      $0xB1,   %%ymm7, %%ymm4         \n" // b5,b4,b7,b6,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm7, %%ymm5         \n" // b6,b7,b4,b5,b2,b3,b0,b1
        "vpermilps      $0x1B,   %%ymm7, %%ymm6         \n" // b7,b6,b5,b4,b3,b2,b1,b0
        "vfmadd231ps    %%ymm7,  %%ymm0, %%ymm12        \n"
        "vfmadd231ps    %%ymm4,  %%ymm0, %%ymm13        \n"
        "vfmadd231ps    %%ymm5,  %%ymm0, %%ymm14        \n"
        "vfmadd231ps    %%ymm6,  %%ymm0, %%ymm15        \n"
        "                                               \n"
        "vperm2f128     $0x1, %%ymm3,%%ymm3,%%ymm7      \n"
        "vmovaps        64(%%rax), %%ymm0               \n"
        "vmovaps        64(%%rbx), %%ymm1               \n"
        "                                               \n" // b0,b1,b2,b3,b4,b5,b6,b7
        "vpermilps      $0xB1,   %%ymm3, %%ymm4         \n" // b1,b0,b3,b2,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm3, %%ymm5         \n" // b2,b3,b0,b1,b6,b7,b4,b5
        "vpermilps      $0x1B,   %%ymm3, %%ymm6         \n" // b3,b2,b1,b0,b7,b6,b5,b4
        "vfmadd231ps    %%ymm3,  %%ymm2, %%ymm8         \n"
        "vfmadd231ps    %%ymm4,  %%ymm2, %%ymm9         \n"
        "vfmadd231ps    %%ymm5,  %%ymm2, %%ymm10        \n"
        "vfmadd231ps    %%ymm6,  %%ymm2, %%ymm11        \n"
        "                                               \n"
        //"vperm2f128     $0x1, %%ymm1,%%ymm1,%%ymm1      \n"
        "                                               \n" // b4,b5,b6,b7,b0,b1,b2,b3
        "vpermilps      $0xB1,   %%ymm7, %%ymm4         \n" // b5,b4,b7,b6,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm7, %%ymm5         \n" // b6,b7,b4,b5,b2,b3,b0,b1
        "vpermilps      $0x1B,   %%ymm7, %%ymm6         \n" // b7,b6,b5,b4,b3,b2,b1,b0
        "vfmadd231ps    %%ymm7,  %%ymm2, %%ymm12        \n"
        "vfmadd231ps    %%ymm4,  %%ymm2, %%ymm13        \n"
        "vfmadd231ps    %%ymm5,  %%ymm2, %%ymm14        \n"
        "vfmadd231ps    %%ymm6,  %%ymm2, %%ymm15        \n"
        "                                               \n"
        "addq           $64,    %%rax                   \n"
        "addq           $64,    %%rbx                   \n"
        "subq           $1,     %%rsi                   \n"
        "jne            .LOOP_ITER                      \n"
        "                                               \n"
        ".LOOP_ITER_LEFT:                               \n"
        "vperm2f128     $0x1, %%ymm1,%%ymm1,%%ymm7      \n" //
        "vmovaps        32(%%rax), %%ymm2               \n" // lsb       ...       msb
        "vmovaps        32(%%rbx), %%ymm3               \n" 
        "                                               \n" // b0,b1,b2,b3,b4,b5,b6,b7
        "vpermilps      $0xB1,   %%ymm1, %%ymm4         \n" // b1,b0,b3,b2,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm1, %%ymm5         \n" // b2,b3,b0,b1,b6,b7,b4,b5
        "vpermilps      $0x1B,   %%ymm1, %%ymm6         \n" // b3,b2,b1,b0,b7,b6,b5,b4
        "vfmadd231ps    %%ymm1,  %%ymm0, %%ymm8         \n"
        "vfmadd231ps    %%ymm4,  %%ymm0, %%ymm9         \n"
        "vfmadd231ps    %%ymm5,  %%ymm0, %%ymm10        \n"
        "vfmadd231ps    %%ymm6,  %%ymm0, %%ymm11        \n"
        "                                               \n"
        //"vperm2f128     $0x1, %%ymm1,%%ymm1,%%ymm1      \n"
        "                                               \n" // b4,b5,b6,b7,b0,b1,b2,b3
        "vpermilps      $0xB1,   %%ymm7, %%ymm4         \n" // b5,b4,b7,b6,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm7, %%ymm5         \n" // b6,b7,b4,b5,b2,b3,b0,b1
        "vpermilps      $0x1B,   %%ymm7, %%ymm6         \n" // b7,b6,b5,b4,b3,b2,b1,b0
        "vfmadd231ps    %%ymm7,  %%ymm0, %%ymm12        \n"
        "vfmadd231ps    %%ymm4,  %%ymm0, %%ymm13        \n"
        "vfmadd231ps    %%ymm5,  %%ymm0, %%ymm14        \n"
        "vfmadd231ps    %%ymm6,  %%ymm0, %%ymm15        \n"
        "                                               \n"
        "vperm2f128     $0x1, %%ymm3,%%ymm3,%%ymm7      \n"
        //"vmovaps        64(%%rax), %%ymm0               \n"
        //"vmovaps        64(%%rbx), %%ymm1               \n"
        "                                               \n" // b0,b1,b2,b3,b4,b5,b6,b7
        "vpermilps      $0xB1,   %%ymm3, %%ymm4         \n" // b1,b0,b3,b2,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm3, %%ymm5         \n" // b2,b3,b0,b1,b6,b7,b4,b5
        "vpermilps      $0x1B,   %%ymm3, %%ymm6         \n" // b3,b2,b1,b0,b7,b6,b5,b4
        "vfmadd231ps    %%ymm3,  %%ymm2, %%ymm8         \n"
        "vfmadd231ps    %%ymm4,  %%ymm2, %%ymm9         \n"
        "vfmadd231ps    %%ymm5,  %%ymm2, %%ymm10        \n"
        "vfmadd231ps    %%ymm6,  %%ymm2, %%ymm11        \n"
        "                                               \n"
        //"vperm2f128     $0x1, %%ymm1,%%ymm1,%%ymm1      \n"
        "                                               \n" // b4,b5,b6,b7,b0,b1,b2,b3
        "vpermilps      $0xB1,   %%ymm7, %%ymm4         \n" // b5,b4,b7,b6,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm7, %%ymm5         \n" // b6,b7,b4,b5,b2,b3,b0,b1
        "vpermilps      $0x1B,   %%ymm7, %%ymm6         \n" // b7,b6,b5,b4,b3,b2,b1,b0
        "vfmadd231ps    %%ymm7,  %%ymm2, %%ymm12        \n"
        "vfmadd231ps    %%ymm4,  %%ymm2, %%ymm13        \n"
        "vfmadd231ps    %%ymm5,  %%ymm2, %%ymm14        \n"
        "vfmadd231ps    %%ymm6,  %%ymm2, %%ymm15        \n"
        ".LOOP_ITER_DONE:                               \n"
        "                                               \n"
        "movq           %1,     %%rsi                   \n" // k_rem
        "testq          %%rsi,  %%rsi                   \n"
        "je             .POST                           \n"
        ".LOOP_REM:                                     \n"
        "vmovaps        (%%rax), %%ymm0                 \n"
        "vmovaps        (%%rbx), %%ymm1                 \n" // b0,b1,b2,b3,b4,b5,b6,b7
        "vpermilps      $0xB1,   %%ymm1, %%ymm2         \n" // b1,b0,b3,b2,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm1, %%ymm3         \n" // b2,b3,b0,b1,b6,b7,b4,b5
        "vpermilps      $0x1B,   %%ymm1, %%ymm4         \n" // b3,b2,b1,b0,b7,b6,b5,b4
        "vfmadd231ps    %%ymm1,  %%ymm0, %%ymm8         \n"
        "vfmadd231ps    %%ymm2,  %%ymm0, %%ymm9         \n"
        "vfmadd231ps    %%ymm3,  %%ymm0, %%ymm10        \n"
        "vfmadd231ps    %%ymm4,  %%ymm0, %%ymm11        \n"
        "                                               \n"
        "vperm2f128     $0x1, %%ymm1,%%ymm1,%%ymm1      \n" // b4,b5,b6,b7,b0,b1,b2,b3
        "vpermilps      $0xB1,   %%ymm1, %%ymm2         \n" // b5,b4,b7,b6,b5,b4,b7,b6
        "vpermilps      $0x4E,   %%ymm1, %%ymm3         \n" // b6,b7,b4,b5,b2,b3,b0,b1
        "vpermilps      $0x1B,   %%ymm1, %%ymm4         \n" // b7,b6,b5,b4,b3,b2,b1,b0
        "vfmadd231ps    %%ymm1,  %%ymm0, %%ymm12        \n"
        "vfmadd231ps    %%ymm2,  %%ymm0, %%ymm13        \n"
        "vfmadd231ps    %%ymm3,  %%ymm0, %%ymm14        \n"
        "vfmadd231ps    %%ymm4,  %%ymm0, %%ymm15        \n"
        "addq           $32,    %%rax                   \n"
        "addq           $32,    %%rbx                   \n"
        "subq           $1,     %%rsi                   \n"
        "jne            .LOOP_REM                       \n"
        "                                               \n"
        ".POST:                                         \n"
        "                                               \n" //C00 C11 C22 C33 C44 C55 C66 C77
                                                            //C01 C10 C23 C32 C45 C54 C67 C76
                                                            //C02 C13 C20 C31 C46 C57 C64 C75
                                                            //C03 C12 C21 C30 C47 C56 C65 C74
                                                            //C04 C15 C26 C37 C40 C51 C62 C73
                                                            //C05 C14 C27 C36 C41 C50 C63 C72
                                                            //C06 C17 C24 C35 C42 C53 C60 C71
                                                            //C07 C16 C25 C34 C43 C52 C61 C70
        "vperm2f128     $0x2,  %%ymm8,  %%ymm12, %%ymm0 \n"
        "vperm2f128     $0x2,  %%ymm9,  %%ymm13, %%ymm1 \n"
        "vperm2f128     $0x2,  %%ymm10, %%ymm14, %%ymm2 \n"
        "vperm2f128     $0x2,  %%ymm11, %%ymm15, %%ymm3 \n"
        "vperm2f128     $0x31, %%ymm8,  %%ymm12, %%ymm4 \n"
        "vperm2f128     $0x31, %%ymm9,  %%ymm13, %%ymm5 \n"
        "vperm2f128     $0x31, %%ymm10, %%ymm14, %%ymm6 \n"
        "vperm2f128     $0x31, %%ymm11, %%ymm15, %%ymm7 \n" //cross-lane
        "                                               \n" //C00 C11 C22 C33 C04 C15 C26 C37
                                                            //C01 C10 C23 C32 C05 C14 C27 C36
                                                            //C02 C13 C20 C31 C06 C17 C24 C35
                                                            //C03 C12 C21 C30 C07 C16 C25 C34
                                                            //C40 C51 C62 C73 C44 C55 C66 C77
                                                            //C41 C50 C63 C72 C45 C54 C67 C76
                                                            //C42 C53 C60 C71 C46 C57 C64 C75
                                                            //C43 C52 C61 C70 C47 C56 C65 C74
        "vshufps        $0x44, %%ymm2,  %%ymm0,  %%ymm8 \n"
        "vshufps        $0x44, %%ymm3,  %%ymm1,  %%ymm9 \n"
        "vshufps        $0xee, %%ymm0,  %%ymm2,  %%ymm10\n"
        "vshufps        $0xee, %%ymm1,  %%ymm3,  %%ymm11\n"
        "vshufps        $0x44, %%ymm6,  %%ymm4,  %%ymm12\n"
        "vshufps        $0x44, %%ymm7,  %%ymm5,  %%ymm13\n"
        "vshufps        $0xee, %%ymm4,  %%ymm6,  %%ymm14\n"
        "vshufps        $0xee, %%ymm5,  %%ymm7,  %%ymm15\n"
        "                                               \n" //C00 C11 C02 C13 C04 C15 C06 C17
                                                            //C01 C10 C03 C12 C05 C14 C07 C16
                                                            //C20 C31 C22 C33 C26 C37 C24 C35
                                                            //C21 C30 C23 C32 C27 C36 C25 C34
                                                            //C40 C51 C42 C53 C44 C55 C46 C57
                                                            //C41 C50 C43 C52 C45 C54 C47 C56
                                                            //C60 C71 C62 C73 C64 C75 C66 C77
                                                            //C61 C70 C63 C72 C65 C74 C67 C76
#if 0
        "vshufps        $0xb1, %%ymm9,  %%ymm9,  %%ymm9 \n"
        "vshufps        $0xb1, %%ymm11, %%ymm11, %%ymm11\n"
        "vshufps        $0xb1, %%ymm13, %%ymm13, %%ymm13\n"
        "vshufps        $0xb1, %%ymm15, %%ymm15, %%ymm15\n"
#endif
        "vpermilps      $0xb1, %%ymm9,   %%ymm9         \n"
        "vpermilps      $0xb1, %%ymm11,  %%ymm11        \n"
        "vpermilps      $0xb1, %%ymm13,  %%ymm13        \n"
        "vpermilps      $0xb1, %%ymm15,  %%ymm15        \n"
        "                                               \n" //C00 C11 C02 C13 C04 C15 C06 C17
                                                            //C10 C01 C12 C03 C15 C05 C16 C07
                                                            //C20 C31 C22 C33 C26 C37 C24 C35
                                                            //C30 C21 C32 C23 C36 C27 C34 C25
                                                            //C40 C51 C42 C53 C44 C55 C46 C57
                                                            //C50 C41 C52 C43 C54 C45 C56 C47
                                                            //C60 C71 C62 C73 C64 C75 C66 C77
                                                            //C70 C61 C72 C63 C74 C65 C76 C67

        "                                               \n"
        "vblendps       $0xaa, %%ymm9,  %%ymm8,  %%ymm0 \n"
        "vblendps       $0x55, %%ymm9,  %%ymm8,  %%ymm1 \n"
        "vblendps       $0xaa, %%ymm11, %%ymm10, %%ymm2 \n"
        "vblendps       $0x55, %%ymm11, %%ymm10, %%ymm3 \n"
        "vblendps       $0xaa, %%ymm13, %%ymm12, %%ymm4 \n"
        "vblendps       $0x55, %%ymm13, %%ymm12, %%ymm5 \n"
        "vblendps       $0xaa, %%ymm15, %%ymm14, %%ymm6 \n"
        "vblendps       $0x55, %%ymm15, %%ymm14, %%ymm7 \n"
        "                                               \n" //C00 ... C07
                                                            //C10 ... C17
                                                            //  ...
        "movq           %4,     %%rax                   \n"
        "movq           %5,     %%rdi                   \n"
        "leaq           (%%rax, %%rdi, 4),  %%rbx       \n"
        "leaq           (%%rbx, %%rdi, 4),  %%rcx       \n"
        "leaq           (%%rcx, %%rdi, 4),  %%rdx       \n"
        "leaq           (%%rdx, %%rdi, 4),  %%r8        \n"
        "leaq           (%%r8,  %%rdi, 4),  %%r9        \n"
        "leaq           (%%r9,  %%rdi, 4),  %%r10       \n"
        "leaq           (%%r10,  %%rdi, 4), %%r11       \n"
        "vaddps         (%%rax),  %%ymm0,  %%ymm0       \n"
        "vaddps         (%%rbx),  %%ymm1,  %%ymm1       \n"
        "vaddps         (%%rcx),  %%ymm2,  %%ymm2       \n"
        "vaddps         (%%rdx),  %%ymm3,  %%ymm3       \n"
        "vaddps         (%%r8 ),  %%ymm4,  %%ymm4       \n"
        "vaddps         (%%r9 ),  %%ymm5,  %%ymm5       \n"
        "vaddps         (%%r10),  %%ymm6,  %%ymm6       \n"
        "vaddps         (%%r11),  %%ymm7,  %%ymm7       \n"
        "vmovaps        %%ymm0,   (%%rax)               \n"
        "vmovaps        %%ymm1,   (%%rbx)               \n"
        "vmovaps        %%ymm2,   (%%rcx)               \n"
        "vmovaps        %%ymm3,   (%%rdx)               \n"
        "vmovaps        %%ymm4,   (%%r8)                \n"
        "vmovaps        %%ymm5,   (%%r9)                \n"
        "vmovaps        %%ymm6,   (%%r10)               \n"
        "vmovaps        %%ymm7,   (%%r11)               \n"
    : // output
    : // input
        "r"(k_itr),     // 0
        "r"(k_rem),     // 1
        "m"(A),         // 2
        "m"(B),         // 3
        "m"(C),         // 4
        "r"(ldc_)       // 5
    : // clobber
        "rax","rbx","rcx","rdx","rsi","rdi",
        "r8","r9","r10","r11",
        "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6",
        "ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13",
        "ymm14","ymm15"
    );
#endif
}