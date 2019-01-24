#ifndef __UTIL_H
#define __UTIL_H

#include <stddef.h>

#ifndef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a):(b) )
#endif

#ifndef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#endif

#ifndef ABS
#define ABS(x) ( ((x)>0) ? (x):(-1*(x)) )
#endif

double current_sec();
void* __aligned_malloc(size_t required_bytes, size_t alignment);
void __aligned_free(void *p);
void rand_vector_f32(float * v, int elem);

static inline unsigned long long sgemm_flop(unsigned long long M, unsigned long long N, unsigned long long K,
    float alpha, float beta)
{
#if 0
    // https://devtalk.nvidia.com/default/topic/482834/how-to-compute-gflops-for-gemm-blas/
    if(alpha == 1.f && beta == 0.f){
        // M*N*K mul, M*N*(K-1) add
        return M*N*(2*K-1);
    }
    if(alpha == 1.f && beta != 0.f){
        // M*N*K mul, M*N*(K-1) add, M*N beta mul, M*N beta add
        return M*N*(2*K + 1);
    }
    if(alpha != 1.f && beta == 0.f){
        // M*N*K mul, M*N*(K-1) add, M*N alpha mul
        return M*N*(2*K);
    }

    // alpha != 1.f, beta != 0.f
    // M*N*K mul, M*N*(K-1) add, M*N alpha mul, M*N beta mul, M*N beta add
    return M*N*(2*K+2);
#endif
    (void)alpha;
    (void)beta;
    return M*N*(K+1)*2;
}


#endif