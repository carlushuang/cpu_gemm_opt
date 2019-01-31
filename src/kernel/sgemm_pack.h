#ifndef __SGEMM_PACK_H
#define __SGEMM_PACK_H


void sgemm_pack(layout_t layout, trans_t trans, identifier_t ident,
    int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest);

#endif