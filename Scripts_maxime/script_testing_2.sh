#!/usr/bin/bash
start=`date +%s`
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 50000000
#~ sudo make -C src/ -j 6
#~ sudo make -j 6

#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1

HOST="gemini-2"
N=40
STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=90 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
