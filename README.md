# gemm_opt

*currently only consider signle cpu core, and sgemm only*

optimize gemm on x86 arch, tested on **Intel(R) Xeon(R) Gold 6142** CPU
* L1d cache:             32K
* L1i cache:             32K
* L2 cache:              1024K
* L3 cache:              22528K
* L1d TLB :              64 entries

most of the size exceed [openblas](https://github.com/xianyi/OpenBLAS)(0.3.4)

![](res/x86_64_compare.jpg)

detail is in [this PDF](res/cpu_gemm.pdf) for how to design the blocking/kernel.

some reference resource on gemm:

* [Anatomy of a high-performance matrix multiplication](https://www.cs.utexas.edu/users/pingali/CS378/2008sp/papers/gotoPaper.pdf)
* [A Family of High-Performance Matrix Multiplication Algorithms](https://www.cs.utexas.edu/~pingali/CS395T/2012sp/papers/MMMvdg.pdf)
* [blislab](https://github.com/flame/blis)

notes: 
* 96% cpu usage perf is achieved by tune the parameter. wait for tuning finish to update following chart
* size below like 768 can't achieve 96%, may need reconsideration on blocking method 
* more fine grain optimization for micro kernel is needed for better perf above 96%, target 98%~99%

```
# 6x16 micro kernel
# need disable intel HT(hyperthread) in BIOS. also, frequency boost is prefered to close.
# ./gemm_driver  -kc 360 -nc 672 -mc 3072
# following char is used by tuned db

cpu:2, freq: 2600.0MHz, theoritical: 81.250 gflops (avx256,fmadd)
l1_size:32K, l2_size:1M, l3_size:22M, page_size:4096, tlb_entry_l1d:64
MC:3072, NC:672, KC:360, MR:6, NR:16
layout:CblasRowMajor, trans_a:CblasNoTrans, trans_b:CblasNoTrans
Considerations:
 L1: MR*KC+NR*KC+MR*NR+NR+MR*NR < L1_size/d_size, lhs:8128, rhs:8192, match?yes
 L2: NC*KC+MR*KC+MR*NC+MR*KC+MR*NC < L2_size/d_size, lhs:254304, rhs:262144, match?yes
 L3: MC*KC+NC*KC+MC*NC+NC*KC+MC*NC < L3_size/d_size, lhs:5718528, rhs:5767168, match?yes
 L1D TLB:
  TA:CEIL(MR*KC*d_size/PAGE_SIZE)+1, 4
  TB:CEIL(NR*KC*d_size/PAGE_SIZE)+1, 7
  TC:up to MR, 6
  TA+2*(TB)+TC < T_entry_total, lhs:24, rhs:64, match?yes

    M    N    K alpha beta   mc    nc   kc  mr  nr   gflops(%)   gflops_ref(%)
   48   48   48  1.0  1.0    108   16   48   6  16  60.13(74.01)  43.05(52.98)  [t]
   96   96   96  1.0  1.0    456   48   96   6  16  67.06(82.53)  56.63(69.70)  [t]
  144  144  144  1.0  1.0    624  448  160   6  16  55.85(68.74)  64.26(79.09)  [t]
  192  192  192  1.0  1.0    432  144  240   6  16  75.79(93.28)  68.14(83.86)  [t]
  240  240  240  1.0  1.0    516  112  320   6  16  74.75(92.00)  69.70(85.78)  [t]
  288  288  288  1.0  1.0    384   80  336   6  16  74.42(91.60)  69.12(85.07)  [t]
  384  384  384  1.0  1.0    432  240  224   6  16  75.11(92.45)  70.93(87.30)  [t]
  480  480  480  1.0  1.0    624  224  256   6  16  76.48(94.13)  73.23(90.13)  [t]
  576  576  576  1.0  1.0    816  240  320   6  16  76.76(94.47)  74.45(91.64)  [t]
  768  768  768  1.0  1.0   1344  384  256   6  16  77.69(95.62)  71.73(88.29)  [t]
  960  960  960  1.0  1.0   1872  336  320   6  16  78.31(96.38)  75.11(92.44)  [t]
 1152 1152 1152  1.0  1.0   2280  384  352   6  16  78.26(96.32)  74.82(92.08)  [t]
 1344 1344 1344  1.0  1.0   1656  448  352   6  16  78.07(96.08)  74.73(91.97)  [t]
 1536 1536 1536  1.0  1.0   2808  512  256   6  16  78.15(96.19)  71.88(88.47)  [t]
 1728 1728 1728  1.0  1.0   2424  448  352   6  16  78.21(96.26)  74.16(91.27)  [t]
 1920 1920 1920  1.0  1.0   3336  384  352   6  16  78.12(96.15)  73.65(90.64)  [t]
 2112 2112 2112  1.0  1.0   2496  448  352   6  16  77.45(95.32)  73.86(90.91)  [t]
 2400 2400 2400  1.0  1.0   2640  448  352   6  16  77.02(94.79)  73.63(90.62)  [t]
 2688 2688 2688  1.0  1.0   2976  448  352   6  16  77.66(95.58)  73.23(90.13)  [t]
 2976 2976 2976  1.0  1.0   3168  576  352   6  16  78.34(96.41)  72.97(89.80)  [t]
 3264 3264 3264  1.0  1.0   3456  640  256   6  16  78.20(96.24)  73.41(90.35)  [t]
 3552 3552 3552  1.0  1.0   3552  576  352   6  16  78.36(96.45)  73.03(89.88)  [t]
 3840 3840 3840  1.0  1.0   3888  512  352   6  16  78.49(96.60)  72.25(88.93)  [t]
 4128 4128 4128  1.0  1.0   4128  448  352   6  16  78.37(96.46)  73.30(90.21)  [t]
 4512 4512 4512  1.0  1.0   2256  576  352   6  16  78.35(96.43)  72.54(89.29)  [t]
 4896 4896 4896  1.0  1.0   4896  384  352   6  16  78.44(96.54)  73.32(90.25)  [t]
 5280 5280 5280  1.0  1.0   2640  576  352   6  16  78.42(96.51)  72.80(89.61)  [t]
 5664 5664 5664  1.0  1.0   3792  512  352   6  16  78.33(96.41)  73.26(90.16)  [t]
 6048 6048 6048  1.0  1.0   3120  576  352   6  16  78.38(96.47)  72.67(89.44)  [t]
 6432 6432 6432  1.0  1.0   3456  576  352   6  16  78.46(96.56)  73.16(90.05)  [t]
 6816 6816 6816  1.0  1.0   3456  576  352   6  16  78.56(96.69)  72.80(89.60)  [t]
 7200 7200 7200  1.0  1.0   3696  512  352   6  16  78.54(96.67)  72.98(89.82)  [t]
 7584 7584 7584  1.0  1.0   3840  512  352   6  16  78.59(96.72)  72.66(89.43)  [t]
 7968 7968 7968  1.0  1.0   4080  448  352   6  16  78.54(96.67)  72.92(89.74)  [t]
 8352 8352 8352  1.0  1.0   4320  448  352   6  16  78.56(96.68)  72.56(89.31)  [t]
 8736 8736 8736  1.0  1.0   4368  448  352   6  16  78.70(96.87)  73.14(90.02)  [t]
 9120 9120 9120  1.0  1.0   3072  576  352   6  16  78.62(96.76)  72.70(89.48)  [t]


```
