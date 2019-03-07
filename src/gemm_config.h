#ifndef __GEMM_CONFIG
#define __GEMM_CONFIG


#define BLOCK_M 516
#define BLOCK_N 4096
#define BLOCK_K 168      // last micro kernel iteratoin
//#define BLOCK_K 384

#define MR 6
#define NR 16


#define L1_SIZE (32*1024)       // l1d size
#define L2_SIZE (1024*1024)
#define L3_SIZE (22528 * 1024)

#define PAGE_SIZE 4096  // 4k page, for most OS/arch
#define CACHELINE_SIZE 64 // 64 byte cache line
#define L1D_TLB_ENTRY 64


#endif