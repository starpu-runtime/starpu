#!/usr/bin/bash
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 50000000

STARPU_SCHED=HFP ORDER_U=1 ./examples/mult/sgemm -xy $((960*5)) -nblocks $((5)) -iter 1

echo "Fin du script"

#~ module load linalg/mkl
#~ libtool --mode=execute gdb --args
#~ STARPU_SCHED=HFP STARPU_NCPU=0 ./examples/mult/sgemm -xy $((960*5)) -nblocks 5 -iter 1

#~ STARPU_SCHED=HFP TASK_STEALING=3 PRINTF=1 MULTIGPU=4 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1

#~ STARPU_SCHED=HFP STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila libtool --mode=execute gdb --args ./examples/mult/sgemm -xy $((960*5)) -nblocks 5 -iter 1
