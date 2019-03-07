#ifndef __SGEMM_PACK_H
#define __SGEMM_PACK_H

#include "../gemm_driver.h"
#include "../util.h"


// https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm-alloc
//extern "C"
//float * sgemm_alloc(identifier_t ident, int m, int n, int k, const gemm_context_t * ctx); 

//extern "C"
//void sgemm_free(float * buf);

extern "C"
void sgemm_pack(layout_t layout, trans_t trans, identifier_t ident,
    int mc, int nc, int kc,
    float alpha, const float * src,
    int ld, float * dest, const gemm_context_t * ctx);

#endif