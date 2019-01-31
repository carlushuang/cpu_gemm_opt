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
// openblas
#include <cblas.h>

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

// https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
class matrix_cvt_t {
public:
    // convert to h,w,h_stride
    std::tuple<size_t, size_t, size_t>
    operator() (size_t row, size_t col, size_t inc_row, size_t inc_col, layout_t layout, trans_t trans){
        if(layout == LAYOUT_ROW_MAJOR){
            if(trans == TRANS_NO_TRANS || trans == TRANS_CONJ_NO_TRANS){
                assert((inc_col == 1) && "inc_col must be 1 in row major no trans layout");
                return std::make_tuple(row, col, inc_row);
            }else{
                assert((inc_row == 1) && "inc_row must be 1 in row major trans layout");
                return std::make_tuple(col, row, inc_col);
            }
        }else{
            if(trans == TRANS_NO_TRANS || trans == TRANS_CONJ_NO_TRANS){
                assert((inc_row == 1) && "inc_row must be 1 in col major no trans layout");
                return std::make_tuple(col, row, inc_col);
            }else{
                assert((inc_col == 1) && "inc_col must be 1 in col major trans layout");
                return std::make_tuple(row, col, inc_row);  
            }
        }
    }
};
class gemm_desc_t {
public:
    // geometry layout desc, row, col, inc_row, inc_col
    typedef std::tuple<size_t, size_t, size_t, size_t> desc_2d_t;
    // memory layout desc, h,w,h_stride, n,c,h,w
    typedef std::tuple<size_t, size_t, size_t> desc_mem_t;

    gemm_desc_t(int m_, int n_, int k_, layout_t layout_, trans_t trans_a_, trans_t trans_b_):
        m(m_),n(n_),k(k_),layout(layout_),trans_a(trans_a_),trans_b(trans_b_)     {}
    desc_2d_t get_a(int lda=0){
        if(layout == LAYOUT_ROW_MAJOR){
            if(trans_a == TRANS_NO_TRANS || trans_a == TRANS_CONJ_NO_TRANS){
                return std::make_tuple(m,k,std::max(k,lda),1);
            }else{
                return std::make_tuple(m,k,1,std::max(m,lda));
            }
        }else{
            if(trans_a == TRANS_NO_TRANS || trans_a == TRANS_CONJ_NO_TRANS){
                return std::make_tuple(m,k,1,std::max(m,lda));
            }else{
                return std::make_tuple(m,k,std::max(k,lda),1);
            }
        }
    }
    desc_2d_t get_b(int ldb=0){
        if(layout == LAYOUT_ROW_MAJOR){
            if(trans_b == TRANS_NO_TRANS || trans_b == TRANS_CONJ_NO_TRANS){
                return std::make_tuple(k,n,std::max(n,ldb),1);
            }else{
                return std::make_tuple(k,n,1,std::max(k,ldb));
            }
        }else{
            if(trans_b == TRANS_NO_TRANS || trans_b == TRANS_CONJ_NO_TRANS){
                return std::make_tuple(k,n,1,std::max(k,ldb));
            }else{
                return std::make_tuple(k,n,std::max(n,ldb),1);
            }
        }
    }
    desc_2d_t get_c(int ldc=0){
        if(layout == LAYOUT_ROW_MAJOR){
            return std::make_tuple(m,n,std::max(n,ldc),1);
        }else{
            return std::make_tuple(m,n,1,std::max(m,ldc));
        }
    }
    desc_mem_t get_mem_a(int lda=0){
        if(layout == LAYOUT_ROW_MAJOR){
            if(trans_a == TRANS_NO_TRANS || trans_a == TRANS_CONJ_NO_TRANS){
                return std::make_tuple(m,k,std::max(k,lda));
            }else{
                return std::make_tuple(k,m,std::max(m,lda));
            }
        }else{
            if(trans_a == TRANS_NO_TRANS || trans_a == TRANS_CONJ_NO_TRANS){
                return std::make_tuple(m,k,std::max(m,lda));
            }else{
                return std::make_tuple(k,m,std::max(k,lda));
            }
        }
    }
    desc_mem_t get_mem_b(int ldb=0){
        if(layout == LAYOUT_ROW_MAJOR){
            if(trans_b == TRANS_NO_TRANS || trans_b == TRANS_CONJ_NO_TRANS){
                return std::make_tuple(k,n,std::max(n,ldb));
            }else{
                return std::make_tuple(n,k,std::max(k,ldb));
            }
        }else{
            if(trans_b == TRANS_NO_TRANS || trans_b == TRANS_CONJ_NO_TRANS){
                return std::make_tuple(k,n,std::max(k,ldb));
            }else{
                return std::make_tuple(n,k,std::max(n,ldb));
            }
        }
    }
    desc_mem_t get_mem_c(int ldc=0){
        if(layout == LAYOUT_ROW_MAJOR){
            return std::make_tuple(m,n,std::max(n,ldc));
        }else{
            return std::make_tuple(n,m,std::max(m,ldc));
        }
    }
private:
    int m, n, k;
    layout_t layout;
    trans_t trans_a;
    trans_t trans_b;
};

class matrix_fp32_t{
public:
    matrix_fp32_t(size_t row_, size_t col_, size_t inc_row_, size_t inc_col_,
        layout_t layout_, trans_t trans_, size_t alignment_)
    {
        this->row = row_;
        this->col = col_;
        this->inc_row = inc_row_;
        this->inc_col = inc_col_;
        this->layout = layout_;
        this->trans = trans_;
        this->alignment = alignment_;
        this->data = (float *) __aligned_malloc(sizeof(float)*elem(), alignment_);
        rand_vector_f32(this->data, elem());

        std::tie(h, w, h_stride) = matrix_cvt_t()(row,col,inc_row,inc_col,layout,trans);
    }
    ~matrix_fp32_t(){
        __aligned_free(this->data);
    }

    void __copy(const matrix_fp32_t & rhs){
        this->row = rhs.row;
        this->col = rhs.col;
        this->inc_row = rhs.inc_row;
        this->inc_col = rhs.inc_col;
        this->layout = rhs.layout;
        this->trans = rhs.trans;
        this->alignment = rhs.alignment;
        this->data = (float *) __aligned_malloc(sizeof(float)*elem(), alignment);
        std::tie(h, w, h_stride) = matrix_cvt_t()(row,col,inc_row,inc_col,layout,trans);
        memcpy(this->data, rhs.data, sizeof(float)*elem());
    }

    matrix_fp32_t(const matrix_fp32_t & rhs){
        __copy(rhs);
    }
    matrix_fp32_t& operator =(const matrix_fp32_t & rhs){
        __copy(rhs);
        return *this;
    }

    size_t elem() const {
          // assume no interleave pattern
        return row*col*inc_row*inc_col;
    }

    float       *data;
    size_t      row;
    size_t      col;
    size_t      inc_row;
    size_t      inc_col;

    layout_t    layout;
    trans_t     trans;
    size_t      alignment;

    // memory layout
    size_t      w;
    size_t      h;
    size_t      h_stride;
};
static inline bool valid_matrix(const matrix_fp32_t *ma, const matrix_fp32_t * mb, float delta){
    int i;
    int errs = 0;
    assert(ma && mb);
    assert(ma->elem() == mb->elem());
    //std::cout<<"ma:"<<ma->elem()<<", mb:"<<mb->elem()<<std::endl;
    for(i=0;i<(int) (ma->elem());i++){
        float d = ma->data[i] - mb->data[i];
        d = ABS(d);
        if(d>delta){
            if(errs<10)
                std::cout<<"["<<i<<"] result diff, left:"<<ma->data[i]<<", right:"<<mb->data[i]<<", delta:"<<d<<std::endl;
            errs++;
        }
    }
    return errs==0;
}

#endif
