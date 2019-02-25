
#include <immintrin.h> // AVX2
#include "../gemm_config.h"
#include "../gemm_driver.h"
#include "sgemm_pack.h"

/*
*    assume A is row major
*    <-  kc ->
*    +-------+      - 
*    |       | mr   ^
*    +-------+      |
*    |       |
*    +-------+
*    |       |      mc
*    +-------+
*    |       |
*    +-------+      |
*    |       |      v
*    +-------+      -
* 
*   pack_A
*
*    mr
*  +---+
*  |   | kc
*  +---+
*  |   | kc
*
* pack_A convert to col major for each tile
*/
static void pack_A(int mc_sz, int kc_sz, const float * A, int lda, float * pack_buf, float alpha){
    int mr,mr_sz, k, i;
    const float * ptr_a = A;
    float * ptr_a_pack = pack_buf;
    (void)alpha;
    for(mr=0;mr<mc_sz;mr+=MR){
        mr_sz = MIN(mc_sz-mr, MR);
        for(k=0;k<kc_sz;k++){
            ptr_a = A + mr*lda + k;
            for(i=0;i<mr_sz;i++){
                *ptr_a_pack++ = *ptr_a;
                ptr_a += lda;
            }
        }
    }
}
static void sgemm_pack_nn_A_mr6(int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest)
{
    unsigned long long k_itr = k/8;
    unsigned long long k_rem = k%8;
    unsigned long long m_itr = m/6;
    unsigned long long m_rem = m%6;
    unsigned long long ld_ = ld;

    asm volatile(
    "movq               %0,     %%rax               \n" // src
    "movq               %1,     %%rbx               \n" // dest
    "movq               %2,     %%rcx               \n" // ld
    "movq               %3,     %%rsi               \n" // k_itr
    "movq               %4,     %%rdi               \n" // k_rem
    "movq               %5,     %%r8                \n" // m_itr
    "movq               %6,     %%r9                \n" // m_rem

    ".LOOP_M_ITR:                                   \n"
    ".LOOP_K_ITR:                                   \n"
    "vmovaps            (%%rax),  %%ymm0            \n"
    "vmovaps            (%%rax,%%rcx,4*1),  %%ymm1  \n"
    "vmovaps            (%%rax,%%rcx,4*2),  %%ymm2  \n"
    "vmovaps            (%%rax,%%rcx,4*3),  %%ymm3  \n"
    "vmovaps            (%%rax,%%rcx,4*4),  %%ymm4  \n"
    "vmovaps            (%%rax,%%rcx,4*5),  %%ymm5  \n"
                                                    // lsb                         msb
                                                    // origin:
                                                    // X00 X01 X02 X03 X04 X05 X06 X07
                                                    // X10 X11 X12 X13 X14 X15 X16 X17
                                                    // X20 X21 X22 X23 X24 X25 X26 X27
                                                    // X30 X31 X32 X33 X34 X35 X36 X37
                                                    // X40 X41 X42 X43 X44 X45 X46 X47
                                                    // X50 X51 X52 X53 X54 X55 X56 X57
                                                    //
                                                    // final:
                                                    // X00 X10 X20 X30 X40 X50  
                                                    // X01 X11 X21 X31 X41 X51  
                                                    // X02 X12 X22 X32 X42 X52  
                                                    // X03 X13 X23 X33 X43 X53  
                                                    // X04 X14 X24 X34 X44 X54  
                                                    // X05 X15 X25 X35 X45 X55  
                                                    // X06 X16 X26 X36 X46 X56  
                                                    // X07 X17 X27 X37 X47 X57
                                                    //
                                                    // reorg:
                                                    // X00 X10 X20 X30 X40 X50 X01 X11
                                                    // X21 X31 X41 X51 X02 X12 X22 X32
                                                    // X42 X52 X03 X13 X23 X33 X43 X53
                                                    // X04 X14 X24 X34 X44 X54 X05 X15
                                                    // X25 X35 X45 X55 X06 X16 X26 X36
                                                    // X46 X56 X07 X17 X27 X37 X47 X57
    "vunpcklps          %%ymm1, %%ymm0, %%ymm6      \n"
    "vunpckhps          %%ymm1, %%ymm0, %%ymm6      \n"
    "vunpcklps          %%ymm3, %%ymm2, %%ymm8      \n"
    "vunpckhps          %%ymm3, %%ymm2, %%ymm9      \n"
    "vunpcklps          %%ymm5, %%ymm4, %%ymm10     \n"
    "vunpckhps          %%ymm5, %%ymm4, %%ymm11     \n"
                                                    // X00 X10 X01 X11 X04 X14 X05 X15
                                                    // X02 X12 X03 X13 X06 X16 X07 X17
                                                    // X20 X30 X21 X31 X24 X34 X25 X35
                                                    // X22 X32 X23 X33 X26 X36 X27 X37
                                                    // X40 X50 X41 X51 X44 X54 X45 X55
                                                    // X42 X52 X43 X53 X46 X56 X47 X57

    "vshufps        $0x44, %%ymm8,  %%ymm6,  %%ymm0 \n"
    "vshufps        $0xee, %%ymm10, %%ymm8,  %%ymm1 \n"
    "vshufps        $0xe4, %%ymm7,  %%ymm11, %%ymm2 \n"
    "vshufps        $0xe4, %%ymm6,  %%ymm10, %%ymm3 \n"
    "vshufps        $0x44, %%ymm9,  %%ymm7,  %%ymm4 \n"
    "vshufps        $0xee, %%ymm11, %%ymm9,  %%ymm5 \n"
                                                    // X00 X10 X20 X30 X04 X14 X24 X34
                                                    // X21 X31 X41 X51 X25 X35 X45 X55
                                                    // X42 X52 X03 X13 X46 X56 X07 X17
                                                    // X40 X50 X01 X11 X44 X54 X05 X15
                                                    // X02 X12 X22 X32 X06 X16 X26 X36
                                                    // X23 X33 X43 X53 X27 X37 X47 X57

    "vperm2f128     $0x20, %%ymm3, %%ymm0, %%ymm6   \n"
    "vperm2f128     $0x20, %%ymm4, %%ymm1, %%ymm7   \n"
    "vperm2f128     $0x20, %%ymm5, %%ymm2, %%ymm8   \n"
    "vperm2f128     $0x31, %%ymm3, %%ymm0, %%ymm9   \n"
    "vperm2f128     $0x31, %%ymm4, %%ymm1, %%ymm10  \n"
    "vperm2f128     $0x31, %%ymm5, %%ymm2, %%ymm11  \n"
                                                    // X00 X10 X20 X30 X40 X50 X01 X11
                                                    // X21 X31 X41 X51 X02 X12 X22 X32
                                                    // X42 X52 X03 X13 X23 X33 X43 X53
                                                    // X04 X14 X24 X34 X44 X54 X05 X15
                                                    // X25 X35 X45 X55 X06 X16 X26 X36
                                                    // X46 X56 X07 X17 X27 X37 X47 X57

    "vmovaps        %%ymm6,     32*0(%%rbx)         \n"
    "vmovaps        %%ymm7,     32*1(%%rbx)         \n"
    "vmovaps        %%ymm8,     32*2(%%rbx)         \n"
    "vmovaps        %%ymm9,     32*3(%%rbx)         \n"
    "vmovaps        %%ymm10,    32*4(%%rbx)         \n"
    "vmovaps        %%ymm11,    32*5(%%rbx)         \n"

    "addq           $6*4,       %%rax               \n"
    "addq           $6*8*4,     %%rbx               \n"
    "subq           $1,         %%rsi               \n"
    "testq          %%rsi,      %%rsi               \n"
    "jne            .LOOP_K_ITR                     \n"

    ".LOOP_K_ITER_DONE:                             \n"

    ".LOOP_K_REM:                                   \n"
    "movq          (%%rax),                 %%r10   \n"
    "movq          (%%rax, %%rcx, 4*1),     %%r11   \n"
    "movq          (%%rax, %%rcx, 4*2),     %%r12   \n"
    "movq          (%%rax, %%rcx, 4*3),     %%r13   \n"
    "movq          (%%rax, %%rcx, 4*4),     %%r14   \n"
    "movq          (%%rax, %%rcx, 4*5),     %%r15   \n"

    "movq           %%r10,        (%%rbx)           \n"
    "movq           %%r11,     4*1(%%rbx)           \n"
    "movq           %%r12,     4*2(%%rbx)           \n"
    "movq           %%r13,     4*3(%%rbx)           \n"
    "movq           %%r14,     4*4(%%rbx)           \n"
    "movq           %%r15,     4*5(%%rbx)           \n"

    "addq           $4,         %%rax               \n"
    "addq           $6*4,       %%rbx               \n"
    "subq           $1,         %%rdi               \n"
    "testq          %%rdi,      %%rdi               \n"
    "jne            .LOOP_K_REM                     \n"

    ".LOOP_K_REM_DONE:                              \n"
    ""
    ""
    : // output
    : // input
        "m"(src),       // 0
        "m"(dest),      // 1
        "m"(ld_),       // 2
        "m"(k_itr),     // 3
        "m"(k_rem),     // 4
        "m"(m_itr),     // 5
        "m"(m_rem)      // 6
    : // clobber
        "rax","rbx","rcx","rdx","rsi","rdi",
        "r8","r9","r10","r11","r12","r13","r14","r15",
        "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6",
        "ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13",
        "ymm14","ymm15"
    );
}

static void sgemm_pack_nn_a(int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest)
{
    if(MR==6)
        return sgemm_pack_nn_A_mr6(m,n,k,alpha,src,ld,dest);
    pack_A(m,k,src,ld,dest,alpha);
}
/*
*  assume B is row major
*  input B block:
*  <-------- nc ------->
*  < nr>
*  +---+---+---+---+---+
*  |   |   |   |   |   |kc
*  |   |   |   |   |   |
*  +---+---+---+---+---+
*    |
*    v
*  pack_B
*
*   nr
*  +---+
*  |   | kc
*  +---+
*  |   |
*  +---+
*
*/
static void pack_B(int nc_sz, int kc_sz, const float * B, int ldb, float * pack_buf, float alpha){
    int k, nr, nr_sz, i;
    const float * ptr_b = B;
    float * ptr_b_pack = pack_buf;
    for(nr=0;nr<nc_sz;nr+=NR){
        nr_sz = MIN(nc_sz-nr, NR);
        for(k=0;k<kc_sz;k++){
            ptr_b = B + k*ldb + nr;
            for(i=0;i<nr_sz;i++){
                *ptr_b_pack++ = *ptr_b++ * alpha;
            }
        }
    }
}
static void sgemm_pack_nn_b_nr8(int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest)
{
    assert(NR==8 && "mx8 kernel pack B");
    (void)m;

    int n_itr = n/8;    // NR
    int n_rem = n%8;

    int k_itr = k/8;    // copy every eight row
    int k_rem = k%8;

    int i,j;

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm_alpha;
    ymm_alpha = _mm256_broadcast_ss(&alpha);

    const float * src_ptr;
    float * dest_ptr = dest;
    for(i=0;i<n_itr;i++){
        src_ptr = src + i*8;
        for(j=0;j<k_itr;j++){
            // load
            ymm0 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            ymm1 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            ymm2 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            ymm3 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            ymm4 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            ymm5 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            ymm6 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            ymm7 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            // scale
            ymm0 = _mm256_mul_ps(ymm0, ymm_alpha);
            ymm1 = _mm256_mul_ps(ymm1, ymm_alpha);
            ymm2 = _mm256_mul_ps(ymm2, ymm_alpha);
            ymm3 = _mm256_mul_ps(ymm3, ymm_alpha);
            ymm4 = _mm256_mul_ps(ymm4, ymm_alpha);
            ymm5 = _mm256_mul_ps(ymm5, ymm_alpha);
            ymm6 = _mm256_mul_ps(ymm6, ymm_alpha);
            ymm7 = _mm256_mul_ps(ymm7, ymm_alpha);
            // store
            _mm256_storeu_ps(dest_ptr, ymm0); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm1); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm2); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm3); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm4); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm5); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm6); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm7); dest_ptr += 8;
        }
        for(j=0;j<k_rem;j++){
            ymm0 = _mm256_loadu_ps(src_ptr); src_ptr += ld;
            ymm0 = _mm256_mul_ps(ymm0, ymm_alpha);
            _mm256_storeu_ps(dest_ptr, ymm0); dest_ptr += 8;
        }
    }
    // final case, copy column one by one
    float k0, k1, k2, k3, k4, k5, k6, k7;
    src = src + n_itr*8;
    dest = dest + k_itr*8;
    for(i=0;i<n_rem;i++){
        src_ptr = src + i;
        dest_ptr = dest + i;
        for(j=0;j<k_itr;j++){
            // load 
            k0 = *src_ptr; src_ptr += ld;
            k1 = *src_ptr; src_ptr += ld;
            k2 = *src_ptr; src_ptr += ld;
            k3 = *src_ptr; src_ptr += ld;
            k4 = *src_ptr; src_ptr += ld;
            k5 = *src_ptr; src_ptr += ld;
            k6 = *src_ptr; src_ptr += ld;
            k7 = *src_ptr; src_ptr += ld;
            // scale
            k0 *= alpha;
            k1 *= alpha;
            k2 *= alpha;
            k3 *= alpha;
            k4 *= alpha;
            k5 *= alpha;
            k6 *= alpha;
            k7 *= alpha;
            // store
            *dest_ptr = k0; dest_ptr+=n_rem;
            *dest_ptr = k1; dest_ptr+=n_rem;
            *dest_ptr = k2; dest_ptr+=n_rem;
            *dest_ptr = k3; dest_ptr+=n_rem;
            *dest_ptr = k4; dest_ptr+=n_rem;
            *dest_ptr = k5; dest_ptr+=n_rem;
            *dest_ptr = k6; dest_ptr+=n_rem;
            *dest_ptr = k7; dest_ptr+=n_rem;
        }
        for(j=0;j<k_rem;j++){
            k0 = *src_ptr; src_ptr += ld;
            k0 *= alpha;
            *dest_ptr = k0; dest_ptr+=n_rem;
        }
    }
}
static void sgemm_pack_nn_b_nr16(int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest)
{
    assert(NR==16 && "mx16 kernel pack B");
    (void)m;

    int n_itr = n/16;    // NR
    int n_rem = n%16;

    int k_itr = k/4;    // copy every 4 row
    int k_rem = k%4;

    int i,j;

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm_alpha;
    ymm_alpha = _mm256_broadcast_ss(&alpha);

    const float * src_ptr;
    float * dest_ptr = dest;
    for(i=0;i<n_itr;i++){
        src_ptr = src + i*16;
        for(j=0;j<k_itr;j++){
            // load
            ymm0 = _mm256_loadu_ps(src_ptr);
            ymm1 = _mm256_loadu_ps(src_ptr+8);
            src_ptr += ld;
            ymm2 = _mm256_loadu_ps(src_ptr);
            ymm3 = _mm256_loadu_ps(src_ptr+8);
            src_ptr += ld;
            ymm4 = _mm256_loadu_ps(src_ptr);
            ymm5 = _mm256_loadu_ps(src_ptr+8);
            src_ptr += ld;
            ymm6 = _mm256_loadu_ps(src_ptr);
            ymm7 = _mm256_loadu_ps(src_ptr+8);
            src_ptr += ld;
            // scale
            ymm0 = _mm256_mul_ps(ymm0, ymm_alpha);
            ymm1 = _mm256_mul_ps(ymm1, ymm_alpha);
            ymm2 = _mm256_mul_ps(ymm2, ymm_alpha);
            ymm3 = _mm256_mul_ps(ymm3, ymm_alpha);
            ymm4 = _mm256_mul_ps(ymm4, ymm_alpha);
            ymm5 = _mm256_mul_ps(ymm5, ymm_alpha);
            ymm6 = _mm256_mul_ps(ymm6, ymm_alpha);
            ymm7 = _mm256_mul_ps(ymm7, ymm_alpha);
            // store
            _mm256_storeu_ps(dest_ptr, ymm0); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm1); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm2); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm3); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm4); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm5); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm6); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm7); dest_ptr += 8;
        }
        for(j=0;j<k_rem;j++){
            ymm0 = _mm256_loadu_ps(src_ptr);
            ymm1 = _mm256_loadu_ps(src_ptr+8); src_ptr += ld;
            ymm0 = _mm256_mul_ps(ymm0, ymm_alpha);
            ymm1 = _mm256_mul_ps(ymm1, ymm_alpha);
            _mm256_storeu_ps(dest_ptr, ymm0); dest_ptr += 8;
            _mm256_storeu_ps(dest_ptr, ymm1); dest_ptr += 8;
        }
    }
    // final case, copy column one by one
    float k0, k1, k2, k3;
    src = src + n_itr*16;
    dest = dest + k_itr*4;
    for(i=0;i<n_rem;i++){
        src_ptr = src + i;
        dest_ptr = dest + i;
        for(j=0;j<k_itr;j++){
            // load 
            k0 = *src_ptr; src_ptr += ld;
            k1 = *src_ptr; src_ptr += ld;
            k2 = *src_ptr; src_ptr += ld;
            k3 = *src_ptr; src_ptr += ld;

            // scale
            k0 *= alpha;
            k1 *= alpha;
            k2 *= alpha;
            k3 *= alpha;

            // store
            *dest_ptr = k0; dest_ptr+=n_rem;
            *dest_ptr = k1; dest_ptr+=n_rem;
            *dest_ptr = k2; dest_ptr+=n_rem;
            *dest_ptr = k3; dest_ptr+=n_rem;
        }
        for(j=0;j<k_rem;j++){
            k0 = *src_ptr; src_ptr += ld;
            k0 *= alpha;
            *dest_ptr = k0; dest_ptr+=n_rem;
        }
    }
}
static void sgemm_pack_nn_b(int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest)
{

    if(NR==8)
        return sgemm_pack_nn_b_nr8(m,n,k,alpha,src,ld,dest);
    if(NR==16)
        return sgemm_pack_nn_b_nr16(m,n,k,alpha,src,ld,dest);
    return pack_B(n,k,src,ld,dest,alpha);
}

//https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm-pack
void sgemm_pack(layout_t layout, trans_t trans, identifier_t ident,
    int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest)
{
    if(layout == LAYOUT_ROW_MAJOR){
        if(trans == TRANS_NO_TRANS || trans == TRANS_CONJ_NO_TRANS){
            if(ident == IDENT_A_MATRIX)
                sgemm_pack_nn_a(m,n,k,alpha,src,ld,dest);
            else
                sgemm_pack_nn_b(m,n,k,alpha,src,ld,dest);
        }
    }else{

    }
}