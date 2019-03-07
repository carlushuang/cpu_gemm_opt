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

#if 0
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
#endif

// C row major, A col major, B row major
extern "C"
void sgemm_macro_kernel_n_tn(
        int    mc,
        int    nc,
        int    kc,
        float  alpha,
        const float * packA,
        const float * packB,
        float  beta,
        float * C,
        int    ldc,
        const gemm_context_t * ctx)
{
    int mr, nr, mr_size, nr_size;
    int mm,nn;
    int page_size;

    mr = ctx->mr;
    nr = ctx->nr;
    page_size = ctx->page_size;

    int offset_a = 0;
    int offset_b = 0;

    for(mm=0; mm<mc; mm += mr){
        mr_size = MIN(mc-mm, mr);
        offset_b = 0;
        for(nn=0; nn<nc; nn += nr){
            nr_size = MIN(nc-nn, nr);
            sgemm_micro_kernel_n_tn(mr_size, nr_size, kc,
                alpha,
                packA + offset_a,
                packB + offset_b,
                beta,
                C+mm*ldc+nn, ldc);
            offset_b += nr*kc;
        }

        //offset_a += page_size / sizeof(float);
        offset_a += mr*kc;
        
    }
}

// C col major, A col major, B row major
extern "C"
void sgemm_macro_kernel_t_tn(
        int    mc,
        int    nc,
        int    kc,
        float  alpha,
        const float * packA,
        const float * packB,
        float  beta,
        float * C,
        int    ldc,
        const gemm_context_t * ctx)
{
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

// C row major, A row major, B row major
static void sgemm_n_nn(
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
#if 0
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
#endif
    int nc, nc_size, kc, kc_size, mc, mc_size;
    int mm, nn, kk;
    int page_size;
    int mr;
    mr = ctx->mr;
    mc = ctx->mc;
    nc = ctx->nc;
    kc = ctx->kc;
    page_size = ctx->page_size;

#if 0
    // alloc A, A should align in one PAGE_SIZE for every mr*kc panel
    int block_bytes = mr*kc*sizeof(float);
    if(block_bytes > page_size){
        std::cerr<<"size for a mr*kc block of A is bigger than page_size, not supported now."<<
        ", mr:"<<mr<<", kc:"<<kc<<", page_size:"<<page_size<<std::endl;
        assert(0);
    }
    int num_pages = (mc-1)/mr + 1;
    float * A_pack = (float*)__aligned_malloc(num_pages * page_size, page_size);
#endif
    // alloc A
    float * A_pack = (float*)__aligned_malloc(mc*kc*sizeof(float), page_size);

    // alloc B
    float * B_pack = (float*)__aligned_malloc(nc*kc*sizeof(float), page_size);

    //printf("[%s] a num tlb:%d, bytes a:%lu, bytes b:%lu\n", __func__, num_pages, num_pages * page_size,nc*kc*sizeof(float) );

    for(mm=0; mm<M; mm += mc){
        mc_size = MIN(M-mm, mc);
        for(kk=0; kk<K; kk += kc){
            kc_size = MIN(K-kk, kc);
            sgemm_pack(LAYOUT_ROW_MAJOR, TRANS_NO_TRANS, IDENT_A_MATRIX,
                    mc_size, 0, kc_size,
                    alpha, A + mm*lda + kk, lda, A_pack, ctx);
            for(nn=0; nn<N; nn += nc){
                nc_size = MIN(N-nn, nc);
                sgemm_pack(LAYOUT_ROW_MAJOR, TRANS_NO_TRANS, IDENT_B_MATRIX,
                    0, nc_size, kc_size,
                    alpha, B + kk*ldb + nn, ldb, B_pack, ctx);

                if( kk==0 )
                    scale_C(mc_size, nc_size, beta, C+mm*ldc+nn, ldc);

                sgemm_macro_kernel_n_tn(mc_size, nc_size, kc_size,
                    alpha, A_pack, B_pack,
                    beta, C+mm*ldc+nn, ldc, ctx);
            }
        }
    }
    __aligned_free(A_pack);
    __aligned_free(B_pack);
}

static void sgemm_n_nt(
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
    // TOTO: implement
}

static void sgemm_n_tn(
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
    // TOTO: implement
}

static void sgemm_n_tt(
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
    // TOTO: implement
}

static void sgemm_t_tt(
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
    // TOTO: implement
}

static void sgemm_t_tn(
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
    // TOTO: implement
}

static void sgemm_t_nt(
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
    // TOTO: implement
}

static void sgemm_t_nn(
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
    // TOTO: implement
}

void cblas_sgemm_opt(layout_t Layout, trans_t Trans_a, trans_t Trans_b,
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
{
    // https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_8
    if(Layout == LAYOUT_ROW_MAJOR){
        if(Trans_a == TRANS_NO_TRANS || Trans_a == TRANS_CONJ_NO_TRANS){
            if(Trans_b == TRANS_NO_TRANS|| Trans_b== TRANS_CONJ_NO_TRANS){
                sgemm_n_nn(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,ctx);
            }else{
                sgemm_n_nt(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,ctx);
            }
        }else{
            if(Trans_b == TRANS_NO_TRANS|| Trans_b== TRANS_CONJ_NO_TRANS){
                sgemm_n_tn(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,ctx);
            }else{
                sgemm_n_tt(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,ctx);
            }
        }
    } else {
        if(Trans_a == TRANS_NO_TRANS || Trans_a == TRANS_CONJ_NO_TRANS){
            if(Trans_b == TRANS_NO_TRANS|| Trans_b== TRANS_CONJ_NO_TRANS){
                sgemm_t_tt(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,ctx);
            }else{
                sgemm_t_tn(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,ctx);
            }
        }else{
            if(Trans_b == TRANS_NO_TRANS|| Trans_b== TRANS_CONJ_NO_TRANS){
                sgemm_t_nt(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,ctx);
            }else{
                sgemm_t_nn(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,ctx);
            }
        }
    }
}