#!/usr/bin/bash
start=`date +%s`
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 50000000
#~ sudo make -C src/ -j 4
sudo make -j 4

#~ ./examples/mult/sgemm -3d -xy $((960*i)) -nblocks $((i)) -nblocksz $((4)) -iter 1
#~ ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1
#~ ./examples/cholesky/cholesky_implicit -size $((960*i)) -nblocks $((i))
#~ ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5

#~ STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0

#### hMETIS #### 
i=5 ; STARPU_NTASKS_THRESHOLD=30 TASK_STEALING=3 HMETIS=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_GENERATE_TRACE=0 PRINTF=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=mst STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1

#~ i=15 ; STARPU_NTASKS_THRESHOLD=30 MULTIGPU=0 HMETIS=2 PRINT3D=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_GENERATE_TRACE=1 PRINTF=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*i)) -nblocks $((i)) -nblocksz $((4)) -iter 1

#### Random taks set ####
#~ STARPU_NTASKS_THRESHOLD=30 PRINTF=1 STARPU_GENERATE_TRACE=0 MULTIGPU=7 TASK_STEALING=3 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=250 STARPU_LIMIT_CUDA_MEM=1050 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 SEED=1 STARPU_HOSTNAME=attila ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5

#~ STARPU_SCHED=HFP MULTIGPU=4 TASK_STEALING=3 PRINTF=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=250 SEED=1 STARPU_LIMIT_CUDA_MEM=1050 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/random_task_graph/random_task_graph -ntasks 500 -ndata 100 -degreemax 2

### Modular heft HFP ###
#~ i=15 ; STARPU_NTASKS_THRESHOLD=30 RANDOM_DATA_ACCESS=0 MULTIGPU=7 PRINTF=1 STARPU_WORKER_STATS=0 TASK_STEALING=3 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_GENERATE_TRACE=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1

#~ STARPU_NTASKS_THRESHOLD=30 PRINTF=1 STARPU_GENERATE_TRACE=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 SEED=1 STARPU_HOSTNAME=attila ./examples/random_task_graph/random_task_graph -ntasks 20 -ndata 100 -degreemax 20
#~ STARPU_NTASKS_THRESHOLD=30 PRINTF=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 SEED=1 STARPU_HOSTNAME=attila ./examples/random_task_graph/random_task_graph -ntasks 500 -ndata 100 -degreemax 20


#~ i=7 ; STARPU_NTASKS_THRESHOLD=30 PRINT3D=0 PRINTF=2 TASK_STEALING=0 INTERLACING=1 MULTIGPU=6 STARPU_WORKER_STATS=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1

#~ ./examples/random_task_graph/random_task_graph

#~ i=30 ; STARPU_NTASKS_THRESHOLD=30 PRINTF=1 MULTIGPU=7 STARPU_WORKER_STATS=1 TASK_STEALING=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_GENERATE_TRACE=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1

#hMETIS#
#~ echo "3 1 20 1 1 1 0 0" > Output_maxime/hMETIS_parameters.txt 
#~ i=10 ; STARPU_NTASKS_THRESHOLD=30 HMETIS=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_GENERATE_TRACE=0 PRINTF=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_WORKER_STATS=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1

#Loop#

#~ libtool --mode=execute gdb --args

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."

#~ 1408.0
