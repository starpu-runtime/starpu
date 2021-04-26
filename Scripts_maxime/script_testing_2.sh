#!/usr/bin/bash
start=`date +%s`
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 50000000

N=$((5))
STARPU_SCHED=HFP TASK_STEALING=0 PRINTF=1 MULTIGPU=4 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
