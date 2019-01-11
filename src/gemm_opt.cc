#include "gemm_driver.h"
#include "kernel/sgemm_micro_kernel.h"

//#define BLOCK_K 128
//#define BLOCK_M 256

#define BLOCK_M 72
#define BLOCK_N 512
#define BLOCK_K 64       // last micro kernel iteratoin

#define MR 8
#define NR 6

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
*    v       +-------+
*  pack_B -> |       | nc
*            +-------+
*                kc
*
* Note, pack_B need convert to col major to let microkernel do well
*/
static void pack_B(int nc_sz, int kc_sz, const float * B, int ldb, float * pack_buf){
    // assume origin B is row major
    int nr,kr,nr_sz;
    //float * ptr[NR];
    const float * ptr = B;
    int i;
    for(nr=0;nr<nc_sz;nr+=NR){
        nr_sz = MIN(nc_sz-nr, NR);
        //for(i=0;i<nr_sz;i++)
        //    ptr[i] = B + i;
        // TODO: fast copy
        for(i=0;i<nr_sz;i++){
            ptr = B+nr+i;
            for(kr=0;kr<kc_sz;kr++){
                *pack_buf++ = *ptr;
                ptr += ldb;
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
*/
static void pack_A(int mc_sz, int kc_sz, const float * A, int lda, float * pack_buf){
    const float * ptr[MR];
    float * pack_ptr[MR];
    int mr, mr_sz, kr;
    int i;
    for(mr=0;mr<mc_sz;mr+=MR){
        mr_sz = MIN(mc_sz-mr,  MR);
        for(i=0;i<mr_sz;i++){
            ptr[i] = A + mr*lda + i*lda;
            pack_ptr[i] = pack_buf + mr*kc_sz + i*kc_sz;
        }
        // TODO: fast copy, maybe copy whole block at once
        for(kr=0;kr<kc_sz;kr++){
            for(i=0;i<mr_sz;i++){
                *pack_ptr[i] = *ptr[i];
                ptr[i] = ptr[i]+1;
                pack_ptr[i] = pack_ptr[i] +1;
            }
        }
    }
}

// scale beta*C outside macro kernel
void sgemm_macro_kernel(
        int    mc,
        int    nc,
        int    kc,
        float  alpha,
        const float * packA,
        const float * packB,
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
            pack_B(nc_size, kc_size, B + kc*ldb + nc, ldb, B_pack);

            for(mc = 0;mc<M;mc += BLOCK_M){
                mc_size = MIN(M-mc, BLOCK_M);
                pack_A(mc_size, kc_size, A + mc*lda + kc, lda, A_pack);

                if( kc==0 )
                    scale_C(mc_size, nc_size, beta, C+mc*ldc+nc, ldc);

                sgemm_macro_kernel(mc_size, nc_size, kc_size,
                    alpha, A_pack, B_pack,
                    C+mc*ldc+nc, ldc);
            }
        }
    }
    __aligned_free(A_pack);
    __aligned_free(B_pack);
}
#if 0
{
    int m, n, k;
    for(m=0;m<M;m+=4){
        for(n=0;n<N;n+=4){
            // TODO: ensure m is divided by 4 !!!
            register float
            c_val_0_0, c_val_0_1, c_val_0_2, c_val_0_3,
            c_val_1_0, c_val_1_1, c_val_1_2, c_val_1_3,
            c_val_2_0, c_val_2_1, c_val_2_2, c_val_2_3,
            c_val_3_0, c_val_3_1, c_val_3_2, c_val_3_3;

            register float
            a_val_0, a_val_1, a_val_2, a_val_3,
            b_val_0, b_val_1, b_val_2, b_val_3;

            c_val_0_0=0; c_val_0_1=0; c_val_0_2=0; c_val_0_3=0;
            c_val_1_0=0; c_val_1_1=0; c_val_1_2=0; c_val_1_3=0;
            c_val_2_0=0; c_val_2_1=0; c_val_2_2=0; c_val_2_3=0;
            c_val_3_0=0; c_val_3_1=0; c_val_3_2=0; c_val_3_3=0;

            float * ptr_a_0 = (float*)&A[(m+0)*lda];
            float * ptr_a_1 = (float*)&A[(m+1)*lda];
            float * ptr_a_2 = (float*)&A[(m+2)*lda];
            float * ptr_a_3 = (float*)&A[(m+3)*lda];

            float * ptr_b_0 = (float*)&B[n];
            float * ptr_b_1 = (float*)&B[n+1];
            float * ptr_b_2 = (float*)&B[n+2];
            float * ptr_b_3 = (float*)&B[n+3];
            for(k=0;k<K;k++) {
                //register float b_val_0, b_val_1, b_val_2, b_val_3;
                //b_val_0 = B[k*ldb+n];
                //b_val_1 = B[k*ldb+n+1];
                //b_val_2 = B[k*ldb+n+2];
                //b_val_3 = B[k*ldb+n+3];
                a_val_0 = *ptr_a_0;
                a_val_1 = *ptr_a_1;
                a_val_2 = *ptr_a_2;
                a_val_3 = *ptr_a_3;

                b_val_0 = *ptr_b_0;
                b_val_1 = *ptr_b_1;
                b_val_2 = *ptr_b_2;
                b_val_3 = *ptr_b_3;

                c_val_0_0 += a_val_0 * b_val_0;
                c_val_1_0 += a_val_1 * b_val_0;
                c_val_2_0 += a_val_2 * b_val_0;
                c_val_3_0 += a_val_3 * b_val_0;

                c_val_0_1 += a_val_0 * b_val_1;
                c_val_1_1 += a_val_1 * b_val_1;
                c_val_2_1 += a_val_2 * b_val_1;
                c_val_3_1 += a_val_3 * b_val_1;

                c_val_0_2 += a_val_0 * b_val_2;
                c_val_1_2 += a_val_1 * b_val_2;
                c_val_2_2 += a_val_2 * b_val_2;
                c_val_3_2 += a_val_3 * b_val_2;

                c_val_0_3 += a_val_0 * b_val_3;
                c_val_1_3 += a_val_1 * b_val_3;
                c_val_2_3 += a_val_2 * b_val_3;
                c_val_3_3 += a_val_3 * b_val_3;

                ptr_a_0++;
                ptr_a_1++;
                ptr_a_2++;
                ptr_a_3++;

                ptr_b_0 += ldb;
                ptr_b_1 += ldb;
                ptr_b_2 += ldb;
                ptr_b_3 += ldb;
            }
            C[(m+0)*ldc+n+0] = c_val_0_0*alpha + C[(m+0)*ldc+n+0]*beta;
            C[(m+1)*ldc+n+0] = c_val_1_0*alpha + C[(m+1)*ldc+n+0]*beta;
            C[(m+2)*ldc+n+0] = c_val_2_0*alpha + C[(m+2)*ldc+n+0]*beta;
            C[(m+3)*ldc+n+0] = c_val_3_0*alpha + C[(m+3)*ldc+n+0]*beta;

            C[(m+0)*ldc+n+1] = c_val_0_1*alpha + C[(m+0)*ldc+n+1]*beta;
            C[(m+1)*ldc+n+1] = c_val_1_1*alpha + C[(m+1)*ldc+n+1]*beta;
            C[(m+2)*ldc+n+1] = c_val_2_1*alpha + C[(m+2)*ldc+n+1]*beta;
            C[(m+3)*ldc+n+1] = c_val_3_1*alpha + C[(m+3)*ldc+n+1]*beta;

            C[(m+0)*ldc+n+2] = c_val_0_2*alpha + C[(m+0)*ldc+n+2]*beta;
            C[(m+1)*ldc+n+2] = c_val_1_2*alpha + C[(m+1)*ldc+n+2]*beta;
            C[(m+2)*ldc+n+2] = c_val_2_2*alpha + C[(m+2)*ldc+n+2]*beta;
            C[(m+3)*ldc+n+2] = c_val_3_2*alpha + C[(m+3)*ldc+n+2]*beta;

            C[(m+0)*ldc+n+3] = c_val_0_3*alpha + C[(m+0)*ldc+n+3]*beta;
            C[(m+1)*ldc+n+3] = c_val_1_3*alpha + C[(m+1)*ldc+n+3]*beta;
            C[(m+2)*ldc+n+3] = c_val_2_3*alpha + C[(m+2)*ldc+n+3]*beta;
            C[(m+3)*ldc+n+3] = c_val_3_3*alpha + C[(m+3)*ldc+n+3]*beta;
        }
    }
}
#endif

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