#!/bin/bash

rm -f generated_model.h

# contrib compact 
./reg_gemm /home/gonnet/These/StarPU-stable/.sampling/starpu_compute_contrib_compact.barracuda.core.debug `wc -l /home/gonnet/These/StarPU-stable/.sampling/starpu_compute_contrib_compact.barracuda.core.debug` 2> /dev/null|sed -s s/GEMM/GEMM_CPU/ >  generated_model.h
./reg_gemm /home/gonnet/These/StarPU-stable/.sampling/starpu_compute_contrib_compact.barracuda.cuda.debug `wc -l /home/gonnet/These/StarPU-stable/.sampling/starpu_compute_contrib_compact.barracuda.cuda.debug` 2> /dev/null|sed -s s/GEMM/GEMM_GPU/ >>  generated_model.h

# strsm

./reg_trsm /home/gonnet/These/StarPU-stable/.sampling/starpu_cblk_strsm.barracuda.cuda.debug `wc -l /home/gonnet/These/StarPU-stable/.sampling/starpu_cblk_strsm.barracuda.cuda.debug` 2> /dev/null|sed -s s/TRSM/TRSM_GPU/ >>  generated_model.h
./reg_trsm /home/gonnet/These/StarPU-stable/.sampling/starpu_cblk_strsm.barracuda.core.debug `wc -l /home/gonnet/These/StarPU-stable/.sampling/starpu_cblk_strsm.barracuda.core.debug` 2> /dev/null|sed -s s/TRSM/TRSM_CPU/ >>  generated_model.h

cat generated_model.h
