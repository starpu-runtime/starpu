#!/usr/bin/bash
#~ start=`date +%s`
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
#~ export STARPU_PERF_MODEL_DIR=/home/gonthier/starpu/tools/perfmodels/sampling
#~ ulimit -S -s 50000000
#~ sudo make -C src/ -j 6
#~ sudo make -j 4

#~ srun --exclusive -C sirocco21 --pty bash Scripts_maxime/task_stealing.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne task_stealing -i

#~ bash Scripts_maxime/task_stealing.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne task_stealing

#~ ./examples/mult/sgemm -3d -xy $((960*i)) -nblocks $((i)) -nblocksz $((4)) -iter 1
#~ ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1
#~ ./examples/cholesky/cholesky_implicit -size $((960*i)) -nblocks $((i))
#~ ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5

#~ libtool --mode=execute gdb --args

#~ STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0

#### hMETIS #### 
#~ i=6 ; STARPU_NTASKS_THRESHOLD=30 TASK_STEALING=3 HMETIS=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_GENERATE_TRACE=0 PRINTF=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=mst STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1

#~ i=15 ; STARPU_NTASKS_THRESHOLD=30 MULTIGPU=0 HMETIS=2 PRINT3D=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_GENERATE_TRACE=1 PRINTF=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*i)) -nblocks $((i)) -nblocksz $((4)) -iter 1

#### Random taks set ####
#~ STARPU_NTASKS_THRESHOLD=30 PRINTF=1 STARPU_GENERATE_TRACE=0 MULTIGPU=7 TASK_STEALING=3 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=250 STARPU_LIMIT_CUDA_MEM=1050 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 SEED=1 STARPU_HOSTNAME=attila ./examples/random_task_graph/random_task_graph -ntasks 10 -ndata 10 -degreemax 5

#~ STARPU_SCHED=HFP MULTIGPU=4 TASK_STEALING=3 PRINTF=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=250 SEED=1 STARPU_LIMIT_CUDA_MEM=1050 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/random_task_graph/random_task_graph -ntasks 500 -ndata 100 -degreemax 2

#### Loop ####
#~ for ((i=1; i<=10; i++))
#~ do
	#~ N=$((5*i))
	i=$((10))
	STARPU_SCHED=eager BELADY=0 ORDER_U=1 PRINTF=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 11
#~ done

#~ end=`date +%s`
#~ runtime=$((end-start))
#~ echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
