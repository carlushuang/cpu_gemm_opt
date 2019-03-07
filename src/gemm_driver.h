#ifndef __GEMM_DRIVER_H
#define __GEMM_DRIVER_H

#include "util.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>
// openblas
#include <cblas.h>

#ifndef CEIL
#define CEIL(value, divider)  (   ((value)-1)/(divider)+1  )
#endif

typedef enum {
    LAYOUT_ROW_MAJOR = 0,
    LAYOUT_COL_MAJOR
}layout_t;

typedef enum {
    TRANS_NO_TRANS = 0,
    TRANS_TRANS,
    TRANS_CONJ_TRANS,
    TRANS_CONJ_NO_TRANS,
}trans_t;

typedef enum {
    IDENT_A_MATRIX = 0,
    IDENT_B_MATRIX
}identifier_t;

// cblas helper function
static inline CBLAS_ORDER to_blas_layout(layout_t layout){
    if(layout == LAYOUT_ROW_MAJOR)
        return CblasRowMajor;
    //if(layout == LAYOUT_COL_MAJOR)
    //TODO: validation
    return CblasColMajor;
}
// cblas helper function
static inline CBLAS_TRANSPOSE to_blas_transpose(trans_t trans){
    switch(trans){
        case  TRANS_NO_TRANS:
            return CblasNoTrans;
        case  TRANS_TRANS:
            return CblasTrans;
        case TRANS_CONJ_TRANS:
            return CblasConjTrans;
        case TRANS_CONJ_NO_TRANS:
            return CblasConjNoTrans;
        default:
            return CblasConjNoTrans;
    }
}
static inline const char * to_layout_str(layout_t layout){
    if(layout == LAYOUT_ROW_MAJOR)
        return "CblasRowMajor";
    if(layout == LAYOUT_COL_MAJOR)
        return "CblasColMajor";
    return "n/a major";
}

static inline const char * to_trans_str(trans_t trans){
    if(trans == TRANS_NO_TRANS)
        return "CblasNoTrans";
    if(trans == TRANS_TRANS)
        return "CblasTrans";
    if(trans == TRANS_CONJ_TRANS)
        return "CblasConjTrans";
    if(trans == TRANS_CONJ_NO_TRANS)
        return "CblasConjNoTrans";
    return "n/a trans";
}

class gemm_context_t {
public:
// matrix descriptors
    layout_t    layout;
    trans_t     trans_a;
    trans_t     trans_b;
    size_t      m;
    size_t      n;
    size_t      k;

    size_t      lda;
    size_t      ldb;
    size_t      ldc;
    double      alpha;
    double      beta;

//
    size_t      alignment;  // used for alloc A/B/C, indeed not so useful

// blocking parameters
    size_t      mc;
    size_t      nc;
    size_t      kc;
    size_t      mr;
    size_t      nr;

// hw parameters
    //size_t      cpu_id;
    size_t      l1_size;
    size_t      l2_size;
    size_t      l3_size;
    size_t      tlb_entry_l1d;
    size_t      cacheline_size;
    size_t      page_size;
    std::vector<int>    cpu_list;

    double      frequency;  // MHz
};

// https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm

class matrix_elem_t {
public:
    size_t operator() (size_t row, size_t col, size_t ldim, layout_t layout, trans_t trans){
        if(layout == LAYOUT_ROW_MAJOR){
            if(trans == TRANS_NO_TRANS || trans == TRANS_CONJ_NO_TRANS){
                assert(ldim>=col);
                return row * ldim;
            }
            else{
                assert(ldim>=row);
                return col * ldim;
            }
        }else{
            if(trans == TRANS_NO_TRANS || trans == TRANS_CONJ_NO_TRANS){
                assert(ldim>=row);
                return col * ldim;
            }
            else{
                assert(ldim>=col);
                return row * ldim;
            }
        }
    }
};

template<typename T>
class matrix_t{
public:
    matrix_t(size_t row_, size_t col_, size_t ldim_,
        layout_t layout_, trans_t trans_, size_t alignment_)
    {
        this->row = row_;
        this->col = col_;
        this->ldim = ldim_;
        // TODO ldim should be multiple of alignment!
        this->layout = layout_;
        this->trans = trans_;
        this->alignment = alignment_;

        size_t elements = matrix_elem_t()(row_, col_, ldim_, layout_, trans_);
        this->data = (T *) __aligned_malloc(sizeof(T)*elements, alignment_);
        rand_vector(this->data, elements);
    }
    ~matrix_t(){
        if(data)
            __aligned_free(this->data);
    }

    size_t dtype_size() const {
        return sizeof(T);
    }
    void __copy(const matrix_t<T> & rhs){
        this->row = rhs.row;
        this->col = rhs.col;
        this->ldim = rhs.ldim;

        this->layout = rhs.layout;
        this->trans = rhs.trans;
        this->alignment = rhs.alignment;
        size_t elements = matrix_elem_t()(rhs.row, rhs.col, rhs.ldim, rhs.layout, rhs.trans);
        this->data = (T *) __aligned_malloc(sizeof(T)*elements, alignment);
        memcpy(this->data, rhs.data, sizeof(T)*elements);
    }

    matrix_t(const matrix_t<T> & rhs){
        __copy(rhs);
    }
    matrix_t& operator =(const matrix_t<T> & rhs){
        __copy(rhs);
        return *this;
    }

    T           *data {nullptr};
    size_t      row;
    size_t      col;
    size_t      ldim;   // leading dimension

    layout_t    layout;
    trans_t     trans;
    size_t      alignment;
};

//typedef matrix_t<float> matrix_fp32_t;
//typedef matrix_t<double> matrix_fp64_t;

#endif
