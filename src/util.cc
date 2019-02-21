#include "util.h"

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <string.h>

double current_sec()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void* __aligned_malloc(size_t required_bytes, size_t alignment)
{
    if( alignment==0 || (alignment & (alignment-1)))    // check pow of 2
        return NULL;
    void* p1; // original block
    void** p2; // aligned block
    int offset = alignment - 1 + sizeof(void*);
    if ((p1 = (void*)malloc(required_bytes + offset)) == NULL)
    {
       return NULL;
    }
    p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void __aligned_free(void *p)
{
    free(((void**)p)[-1]);
}

void rand_vector_f32(float * v, int elem) {
    int i;

    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }

    for(i=0;i<elem;i++){
        v[i] = ((float)(rand() % 100)) / 100.0f;
    }
}

#ifdef __x86_64
#define __cpuid(eax,ebx,ecx,edx)    \
    asm volatile(                   \
        "  xchgq  %%rbx,%q1\n"      \
        "  cpuid\n"                 \
        "  xchgq  %%rbx,%q1"        \
        : "=a"(eax), "=r" (ebx), "=c"(ecx), "=d"(edx) \
        : "a"(eax), "c"(ecx))
#else
// TODO 32bit calling convention
#endif
typedef union{
    struct {
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
    };
    uint8_t str[12];
}vendor_str_t;
// must at least 12 char plus 1 \0
void cpuid_vendor_str(char * vendor_str){
    uint32_t eax,ebx,ecx,edx;
    eax = 0;
    __cpuid(eax,ebx,ecx,edx);
    //a twelve-character ASCII string stored in EBX, EDX, ECX

    vendor_str_t vs;
    vs.ebx = ebx;
    vs.edx = edx;
    vs.ecx = ecx;
    strncpy(vendor_str, (const char*)vs.str, 12);
}
int cpuid_support_avx(){
/*
1) Detect CPUID.1:ECX.OSXSAVE[bit 27] = 1 (XGETBV enabled for application use 1 )
2) Issue XGETBV and verify that XCR0[2:1] = ‘11b’ (XMM state and YMM state are enabled by OS).
3) detect CPUID.1:ECX.AVX[bit 28] = 1 (AVX instructions supported).
*/
    uint32_t eax,ebx,ecx,edx;

    // step 1
    eax = 1;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t osxsave = ecx & (1<<27);
    if(!osxsave)
        return 0;

    // step 2
    ecx = 0;        // request XCR0
    asm volatile("xgetbv \n" : "=a"(eax), "=d"(edx): "c"(ecx));
    uint64_t xcr0 = ((uint64_t)edx<<32) | eax;
    if(!((xcr0 & 0x6) == 0x6))
        return 0;

    // step 3
    eax = 1;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t avx = ecx & (1<<28);
    return avx? 1:0;
}

int cpuid_support_avx2(){
    uint32_t eax,ebx,ecx,edx;
    // step 1
    eax = 1;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t osxsave = ecx & (1<<27);
    if(!osxsave)
        return 0;

    // step 2
    ecx = 0;        // request XCR0
    asm volatile("xgetbv \n" : "=a"(eax), "=d"(edx): "c"(ecx));
    uint64_t xcr0 = ((uint64_t)edx<<32) | eax;
    if(!((xcr0 & 0x6) == 0x6))
        return 0;

    // step 3
    eax = 7;
    ecx = 0;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t avx2 = ebx & (1<<5);
    return avx2? 1:0;
}

// DETECTION OF 512-BIT INSTRUCTION GROUPS OF INTEL AVX-512 FAMILY
// foundation
int cpuid_support_avx512_f(){
/*
1. Detect CPUID.1:ECX.OSXSAVE[bit 27] = 1 (XGETBV enabled for application use 1 ).
2. Execute XGETBV and verify that XCR0[7:5] = ‘111b’ (OPMASK state, upper 256-bit of ZMM0-ZMM15 and
ZMM16-ZMM31 state are enabled by OS) and that XCR0[2:1] = ‘11b’ (XMM state and YMM state are enabled by
OS).
3. Detect CPUID.0x7.0:EBX.AVX512F[bit 16] = 1.
*/
    uint32_t eax,ebx,ecx,edx;

    // step 1
    eax = 1;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t osxsave = ecx & (1<<27);
    if(!osxsave)
        return 0;

    // step 2
    ecx = 0;        // request XCR0
    asm volatile("xgetbv \n" : "=a"(eax), "=d"(edx): "c"(ecx));
    uint64_t xcr0 = ((uint64_t)edx<<32) | eax;
    if(!((xcr0 & 0xe) == 0xe) || !((xcr0 & 0x6) == 0x6))
        return 0;

    // step 3
    eax = 7;
    ecx = 0;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t avx512f = ebx & (1<<16);
    return avx512f ? 1:0;
}
// prefetch, cpuid_support_avx512_f() must check first
int cpuid_support_avx512_pf(){
    uint32_t eax,ebx,ecx,edx;
    eax = 7;
    ecx = 0;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t avx512pf = ebx & (1<<26);
    return avx512pf ? 1:0;
}

// exponent & reciprocal, cpuid_support_avx512_f() must check first
int cpuid_support_avx512_er(){
    uint32_t eax,ebx,ecx,edx;
    eax = 7;
    ecx = 0;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t avx512er = ebx & (1<<27);
    return avx512er ? 1:0;
}
// conflict detection, cpuid_support_avx512_f() must check first
int cpuid_support_avx512_cd(){
    uint32_t eax,ebx,ecx,edx;
    eax = 7;
    ecx = 0;
    __cpuid(eax,ebx,ecx,edx);
    uint32_t avx512cd = ebx & (1<<28);
    return avx512cd ? 1:0;
}

