#ifndef __SGEMM_PACK_H
#define __SGEMM_PACK_H



// scoped symbol
#define SSYM (symbol_name)   __func__ ## sym_name

extern "C"
void sgemm_pack(layout_t layout, trans_t trans, identifier_t ident,
    int m, int n, int k,
    float alpha, const float * src,
    int ld, float * dest);

#endif