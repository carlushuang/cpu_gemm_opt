#ifndef __UTIL_H
#define __UTIL_H

#include <stddef.h>
#include <vector>

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

template<typename T>
void rand_vector(T* v, int elem);


// TODO: need disable Intel HT(HyperThread). with HT on, seems thread use both virtual thread
void set_current_affinity(const std::vector<int> & affinity);
void get_current_affinity(std::vector<int> & affinity);
int get_current_cpu();

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

void cpuid_vendor_str(char * vendor_str);
/* F, CD, ER, PF
* Introduced with Xeon Phi x200 (Knights Landing) and Xeon E5-26xx V5 (Skylake EP/EX "Purley", expected in H2 2017), 
* with the last two (ER and PF) being specific to Knights Landing.
*
*/
int cpuid_support_avx();
int cpuid_support_avx2();
int cpuid_support_avx512_f();
int cpuid_support_avx512_pf();
int cpuid_support_avx512_er();
int cpuid_support_avx512_cd();
#endif