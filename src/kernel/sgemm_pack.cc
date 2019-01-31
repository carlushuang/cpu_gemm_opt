
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

static void sgemm_pack_nn_a(int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest)
{
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
static void sgemm_pack_nn_b(int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest)
{
    //pack_B(n,k,src,ld,dest,alpha);
    assert(MR==4 && NR==8 && "4x8 kernel pack B");
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