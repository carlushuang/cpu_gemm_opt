#include "gemm_driver.h"
#include "kernel/sgemm_micro_kernel.h"
#include "kernel/sgemm_pack.h"
#include "gemm_config.h"

//#define BLOCK_K 128
//#define BLOCK_M 256

#ifndef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a):(b) )
#endif

#ifndef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#endif

#define ALIGN_SIZE 32


extern "C"
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
#if 0
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
#endif
    int mr, nr, mr_sz, nr_sz;
    for(nr=0;nr<nc;nr+=NR){
        nr_sz = MIN(nc-nr, NR);
        for(mr=0;mr<mc;mr+=MR){
            // mr loop, keep loop over A mr block, hence let micro kernel assumption fit what blis say
            // here B block keep the same for every iter of mr, let later B fit in L1
            mr_sz = MIN(mc-mr, MR);
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
            //pack_B(nc_size, kc_size, B + kc*ldb + nc, ldb, B_pack, alpha);
            sgemm_pack(LAYOUT_ROW_MAJOR, TRANS_NO_TRANS, IDENT_B_MATRIX,
                0, nc_size, kc_size,
                alpha, B + kc*ldb + nc, ldb, B_pack);

            for(mc = 0;mc<M;mc += BLOCK_M){
                mc_size = MIN(M-mc, BLOCK_M);
                //pack_A(mc_size, kc_size, A + mc*lda + kc, lda, A_pack, alpha);
                sgemm_pack(LAYOUT_ROW_MAJOR, TRANS_NO_TRANS, IDENT_A_MATRIX,
                    mc_size, 0, kc_size,
                    alpha, A + mc*lda + kc, lda, A_pack);

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