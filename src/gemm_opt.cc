#include "gemm_driver.h"

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>  // FMA

#define BLOCK_K 128
#define BLOCK_M 256
typedef union{
    __m128 v;
    float  f[4];
}v4f_t;

static void inner_pack_AB( int M, int N, int K,  float alpha,
                                        float *A, int lda, 
                                        float *B, int ldb,
                                        float beta,
                                        float *C, int ldc ){
    static float A_pack[BLOCK_M*BLOCK_K] __attribute__ ((aligned(16)));
    float B_pack[K*N] __attribute__ ((aligned(16)));
    //__builtin_memcpy(A_pack, A, M*K*sizeof(float));
    int m, n, k;

    for(n=0;n<N;n+=4) {
        // pack B
        int ik;
        for(ik=0;ik<K;ik++){
            B_pack[ik*N+n+0] = B[ik*ldb+n+0];
            B_pack[ik*N+n+1] = B[ik*ldb+n+1];
            B_pack[ik*N+n+2] = B[ik*ldb+n+2];
            B_pack[ik*N+n+3] = B[ik*ldb+n+3];
        }
        for(m=0;m<M;m+=4) {
            v4f_t c_val_r0, c_val_r1, c_val_r2, c_val_r3;
            if( n == 0 ) {
                // pack A
                __builtin_memcpy(A_pack+m*BLOCK_K           , A+(m+0)*lda, sizeof(float)*K);
                __builtin_memcpy(A_pack+m*BLOCK_K+BLOCK_K   , A+(m+1)*lda, sizeof(float)*K);
                __builtin_memcpy(A_pack+m*BLOCK_K+BLOCK_K*2 , A+(m+2)*lda, sizeof(float)*K);
                __builtin_memcpy(A_pack+m*BLOCK_K+BLOCK_K*3 , A+(m+3)*lda, sizeof(float)*K);
            }

            c_val_r0.v = _mm_setzero_ps();
            c_val_r1.v = _mm_setzero_ps();
            c_val_r2.v = _mm_setzero_ps();
            c_val_r3.v = _mm_setzero_ps();

            float * ptr_a_0 = (float*)&A_pack[m*BLOCK_K+0*BLOCK_K];
            float * ptr_a_1 = (float*)&A_pack[m*BLOCK_K+1*BLOCK_K];
            float * ptr_a_2 = (float*)&A_pack[m*BLOCK_K+2*BLOCK_K];
            float * ptr_a_3 = (float*)&A_pack[m*BLOCK_K+3*BLOCK_K];

            float * ptr_b_0 = (float*)&B_pack[n];

            v4f_t a_val_0, a_val_1, a_val_2, a_val_3, b_val;
            for(k=0;k<K;k++) {

                a_val_0.v  = _mm_load_ps1(ptr_a_0);
                a_val_1.v  = _mm_load_ps1(ptr_a_1);
                a_val_2.v  = _mm_load_ps1(ptr_a_2);
                a_val_3.v  = _mm_load_ps1(ptr_a_3);

                // TODO: ptr must be 16 byte alignment
                b_val.v    = _mm_load_ps(ptr_b_0);

                c_val_r0.v = _mm_fmadd_ps(a_val_0.v, b_val.v, c_val_r0.v);

                //a_val.v    = _mm_set_ps(*ptr_a_1, *ptr_a_1, *ptr_a_1, *ptr_a_1);
                c_val_r1.v = _mm_fmadd_ps(a_val_1.v, b_val.v, c_val_r1.v);

                //a_val.v    = _mm_set_ps(*ptr_a_2, *ptr_a_2, *ptr_a_2, *ptr_a_2);
                c_val_r2.v = _mm_fmadd_ps(a_val_2.v, b_val.v, c_val_r2.v);

                //a_val.v    = _mm_set_ps(*ptr_a_3, *ptr_a_3, *ptr_a_3, *ptr_a_3);
                c_val_r3.v = _mm_fmadd_ps(a_val_3.v, b_val.v, c_val_r3.v);

                ptr_a_0++;
                ptr_a_1++;
                ptr_a_2++;
                ptr_a_3++;

                ptr_b_0 += N;
            }
            C[(m+0)*ldc+n+0] = c_val_r0.f[0]*alpha + C[(m+0)*ldc+n+0]*beta;
            C[(m+1)*ldc+n+0] = c_val_r1.f[0]*alpha + C[(m+1)*ldc+n+0]*beta;
            C[(m+2)*ldc+n+0] = c_val_r2.f[0]*alpha + C[(m+2)*ldc+n+0]*beta;
            C[(m+3)*ldc+n+0] = c_val_r3.f[0]*alpha + C[(m+3)*ldc+n+0]*beta;

            C[(m+0)*ldc+n+1] = c_val_r0.f[1]*alpha + C[(m+0)*ldc+n+1]*beta;
            C[(m+1)*ldc+n+1] = c_val_r1.f[1]*alpha + C[(m+1)*ldc+n+1]*beta;
            C[(m+2)*ldc+n+1] = c_val_r2.f[1]*alpha + C[(m+2)*ldc+n+1]*beta;
            C[(m+3)*ldc+n+1] = c_val_r3.f[1]*alpha + C[(m+3)*ldc+n+1]*beta;

            C[(m+0)*ldc+n+2] = c_val_r0.f[2]*alpha + C[(m+0)*ldc+n+2]*beta;
            C[(m+1)*ldc+n+2] = c_val_r1.f[2]*alpha + C[(m+1)*ldc+n+2]*beta;
            C[(m+2)*ldc+n+2] = c_val_r2.f[2]*alpha + C[(m+2)*ldc+n+2]*beta;
            C[(m+3)*ldc+n+2] = c_val_r3.f[2]*alpha + C[(m+3)*ldc+n+2]*beta;

            C[(m+0)*ldc+n+3] = c_val_r0.f[3]*alpha + C[(m+0)*ldc+n+3]*beta;
            C[(m+1)*ldc+n+3] = c_val_r1.f[3]*alpha + C[(m+1)*ldc+n+3]*beta;
            C[(m+2)*ldc+n+3] = c_val_r2.f[3]*alpha + C[(m+2)*ldc+n+3]*beta;
            C[(m+3)*ldc+n+3] = c_val_r3.f[3]*alpha + C[(m+3)*ldc+n+3]*beta;
        }
    }
}

static void sgemm_nn(int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc)
#if 0
{
    if(beta != 1.0f  && beta != 0){
        int i, j;
        for(j=0;j<M;j++){
            for(i=0;i<N;i++){
                int idx = j*ldc+i;
                C[idx] *= beta;
            }
        }
    }
    int im, ik;
    for(im = 0; im < M; im+=BLOCK_M){
        for(ik = 0; ik < K; ik += BLOCK_K){
            inner_pack_AB( MIN(M-im, BLOCK_M), N, MIN(K-ik, BLOCK_K), alpha,
                            (float*)&A[im*lda + ik], lda, 
                            (float*)&B[ik*ldb], ldb,
                            1.0f,                       // force beta tobe 1
                            &C[im*ldc], ldc); 
        }
    }
}
#endif
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