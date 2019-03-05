#!/bin/sh

OPENBLAS_DIR=/opt/OpenBLAS/
CC=/opt/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang++
SRC="gemm_driver.cc gemm_opt.cc util.cc kernel/sgemm_c.cc kernel/sgemm_pack.cc  \
    kernel/sgemm_asm_4x8.cc kernel/sgemm_asm_8x8.cc kernel/sgemm_asm_4x16.cc \
    kernel/sgemm_asm_6x16.cc"
CXXFLAGS=" -pthread -std=c++11 -Wall -O3 -I${OPENBLAS_DIR}/include/ -m64 -mfma -msse -msse2"
CXXFLAGS="${CXXFLAGS} -g "
LDFLAGS=" -L${OPENBLAS_DIR}/lib -lopenblas -lm -Wl,-rpath,${OPENBLAS_DIR}/lib"
TARGET=gemm_driver

rm -rf $TARGET
$CC $CXXFLAGS $SRC $LDFLAGS -o $TARGET
