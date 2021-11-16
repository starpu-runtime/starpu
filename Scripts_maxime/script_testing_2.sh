#!/usr/bin/bash
start=`date +%s`
ulimit -S -s 5000000

N=35

COUNT_DO_SCHEDULE=1 PRINT_TIME=1 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_BUS_STATS=1 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."



#~ # size	ms	GFlops
#~ 24000	1119	4117.7

#~ #---------------------
#~ Data transfer stats:
	#~ NUMA 0 -> CUDA 0	2.1046 GB	28.7782 MB/s	(transfers : 613 - avg 3.5156 MB)
	#~ CUDA 0 -> NUMA 0	0.0000 GB	0.0000 MB/s	(transfers : 0 - avg -nan MB)
#~ Total transfers: 2.1046 GB
#~ #---------------------
#~ Fin du script, l'execution a durée 1 min 16 sec.
