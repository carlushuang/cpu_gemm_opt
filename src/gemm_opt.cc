#include "gemm_driver.h"
#include "kernel/sgemm_micro_kernel.h"

//#define BLOCK_K 128
//#define BLOCK_M 256

#define BLOCK_M 72
#define BLOCK_N 512
#define BLOCK_K 64       // last micro kernel iteratoin

#define MR 8
#define NR 4

#ifndef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a):(b) )
#endif

#ifndef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#endif

#define ALIGN_SIZE 32

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
// pre scale alpha here
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

void sgemm_macro_kernel( 
        int    mc,
        int    nc,
        int    kc,
        float  alpha,
        const float * packA,
        const float * packB,
        float  beta,
        float * C,
        int    ldc )
{
    int mr, nr, mr_sz, nr_sz;
    for(mr=0;mr<mc;mr+=MR){
        mr_sz = MIN(mc-mr, MR);
        for(nr=0;nr<nc;nr+=NR){
            nr_sz = MIN(nc-nr, NR);
            sgemm_micro_kernel(mr_sz, nr_sz, kc,
                alpha,
                packA + mr*kc,
                packB + nr*kc,
                beta,
                C+mr*ldc+nr, ldc);
        }
    }
}

static void scale_C(int mc, int nc, float beta, float * C, int ldc){
    float * c_itr = C;
    if(beta == 1.f){
        ;
    }else if(beta == 0.f){
        int i,j;
        for(j=0;j<mc;j++){
            c_itr = C;
            for(i=0;i<nc;i++){
                *c_itr++ = 0.f;
            }
            C += ldc;
        }
    }else{
        int i,j;
        for(j=0;j<mc;j++){
            c_itr = C;
            for(i=0;i<nc;i++){
                *c_itr = beta * (*c_itr);
                c_itr++;
            }
            C += ldc;
        }
    }
}

// assume row major
static void sgemm_nn(int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc)
{
    int nc, nc_size, kc, kc_size, mc, mc_size;
    // blis algorithm
    float * A_pack = (float*)__aligned_malloc(BLOCK_M*BLOCK_K*sizeof(float), ALIGN_SIZE);
    float * B_pack = (float*)__aligned_malloc(BLOCK_K*BLOCK_N*sizeof(float), ALIGN_SIZE);

    for(nc =0;nc < N;nc+= BLOCK_N){
        nc_size = MIN(N-nc, BLOCK_N);
        for(kc = 0;kc<K; kc+= BLOCK_K){
            kc_size = MIN(K-kc, BLOCK_K);
            pack_B(nc_size, kc_size, B + kc*ldb + nc, ldb, B_pack, alpha);

            for(mc = 0;mc<M;mc += BLOCK_M){
                mc_size = MIN(M-mc, BLOCK_M);
                pack_A(mc_size, kc_size, A + mc*lda + kc, lda, A_pack, alpha);

                if( kc==0 )
                    scale_C(mc_size, nc_size, beta, C+mc*ldc+nc, ldc);

                sgemm_macro_kernel(mc_size, nc_size, kc_size,
                    alpha, A_pack, B_pack,
                    beta, C+mc*ldc+nc, ldc);
            }
        }
    }
    __aligned_free(A_pack);
    __aligned_free(B_pack);
}

void cblas_sgemm_opt(layout_t Layout, trans_t Trans_a, trans_t Trans_b,
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc)
{
    // https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_8
    if(Layout == LAYOUT_ROW_MAJOR){
        if(Trans_a == TRANS_NO_TRANS || Trans_a == TRANS_CONJ_NO_TRANS){
            if(Trans_b == TRANS_NO_TRANS|| Trans_b== TRANS_CONJ_NO_TRANS){
                sgemm_nn(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
            }else{

            }
        }else{

        }
    } else {
        //
    }
}