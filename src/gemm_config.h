#ifndef __GEMM_CONFIG
#define __GEMM_CONFIG


#define BLOCK_M 516
#define BLOCK_N 4096
#define BLOCK_K 256      // last micro kernel iteratoin
//#define BLOCK_K 384

#define MR 6
#define NR 16

#define PAGE_SIZE 4096  // 4k page, for most OS/arch

#endif