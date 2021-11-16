#!/usr/bin/bash
start=`date +%s`
ulimit -S -s 5000000

N=25

COUNT_DO_SCHEDULE=0 PRINT_TIME=1 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_BUS_STATS=1 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
