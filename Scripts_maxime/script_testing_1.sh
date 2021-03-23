#!/usr/bin/bash
start=`date +%s`
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 50000000

#~ i=15 ; STARPU_NTASKS_THRESHOLD=30 PRINTF=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_TASK_PROGRESS=0 BELADY=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*i)) -nblocks $((i)) -nblocksz $((4)) -iter 1

#~ i=50 ; STARPU_NTASKS_THRESHOLD=30 PRINTF=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=mst STARPU_TASK_PROGRESS=1 BELADY=0 ORDER_U=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1

#~ i=30 ; BELADY=1 PRINTF=1 STARPU_SCHED=mst ORDER_U=0 STARPU_NTASKS_THRESHOLD=30 STARPU_TASK_PROGRESS=1 STARPU_CUDA_PIPELINE=4 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*i)) -nblocks $((i)) | tail -n 1

#### Multi GPU ####
#~ i=10 ; STARPU_NTASKS_THRESHOLD=30 PRINTF=1 MULTIGPU=2 STARPU_WORKER_STATS=1 STARPU_BUS_STATS=1 STARPU_GENERATE_TRACE=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_TASK_PROGRESS=0 BELADY=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*i)) -nblocks $((i)) -nblocksz $((4)) -iter 1
i=15 ; STARPU_NTASKS_THRESHOLD=30 PRINTF=1 STARPU_WORKER_STATS=1 STARPU_BUS_STATS=1 MULTIGPU=2 STARPU_GENERATE_TRACE=1 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_TASK_PROGRESS=0 STARPU_GENERATE_TRACE=1 BELADY=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1
#~ i=20 ; STARPU_NTASKS_THRESHOLD=30 PRINTF=1 MULTIGPU=2 STARPU_GENERATE_TRACE=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_TASK_PROGRESS=0 BELADY=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1
#~ i=20 ; STARPU_NTASKS_THRESHOLD=30 PRINTF=1 MULTIGPU=3 STARPU_GENERATE_TRACE=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_TASK_PROGRESS=0 BELADY=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*i)) -nblocks $((i)) -iter 1


#~ i=44 ;
#~ STARPU_SCHED=dmdar STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*i)) -nblocks $((i)) -no-prio

#~ STARPU_SCHED=modular-heft STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_SCHED_SORTED_ABOVE=0 STARPU_SCHED_SORTED_BELOW=0 STARPU_NTASKS_THRESHOLD=0 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*i)) -nblocks $((i)) -no-prio


#~ libtool --mode=execute gdb --args

end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."

#~ #---------------------
#~ Worker stats:
#~ CUDA 0.0 (Simgrid 0.2 GiB)      
	#~ 580 task(s)
	#~ total: 4469.00 ms executing: 1631.44 ms sleeping: 1341.85 ms overhead 1495.71 ms
	#~ 229.647334 GFlop/s

#~ CUDA 1.0 (Simgrid 0.2 GiB)      
	#~ 557 task(s)
	#~ total: 4469.00 ms executing: 1566.27 ms sleeping: 1315.89 ms overhead 1586.85 ms
	#~ 220.540629 GFlop/s

#~ CUDA 2.0 (Simgrid 0.2 GiB)      
	#~ 463 task(s)
	#~ total: 4469.00 ms executing: 1303.52 ms sleeping: 1294.71 ms overhead 1870.77 ms
	#~ 183.321924 GFlop/s

#~ #---------------------

#~ #---------------------
#~ Worker stats:
#~ CUDA 0.0 (Simgrid 0.2 GiB)      
	#~ 525 task(s)
	#~ total: 5445.73 ms executing: 1477.74 ms sleeping: 1862.91 ms overhead 2105.08 ms
	#~ 170.587222 GFlop/s

#~ CUDA 1.0 (Simgrid 0.2 GiB)      
	#~ 460 task(s)
	#~ total: 5445.73 ms executing: 1294.35 ms sleeping: 1925.43 ms overhead 2225.96 ms
	#~ 149.466899 GFlop/s

#~ CUDA 2.0 (Simgrid 0.2 GiB)      
	#~ 615 task(s)
	#~ total: 5445.73 ms executing: 1731.08 ms sleeping: 1294.52 ms overhead 2420.13 ms
	#~ 199.830745 GFlop/s

#~ #---------------------

#~ #---------------------
#~ Worker stats:
#~ CUDA 0.0 (Simgrid 0.2 GiB)      
	#~ 553 task(s)
	#~ total: 5293.23 ms executing: 1556.17 ms sleeping: 1520.60 ms overhead 2216.46 ms
	#~ 184.862165 GFlop/s

#~ CUDA 1.0 (Simgrid 0.2 GiB)      
	#~ 553 task(s)
	#~ total: 5293.23 ms executing: 1555.58 ms sleeping: 1437.58 ms overhead 2300.07 ms
	#~ 184.862165 GFlop/s

#~ CUDA 2.0 (Simgrid 0.2 GiB)      
	#~ 494 task(s)
	#~ total: 5293.23 ms executing: 1390.87 ms sleeping: 1294.52 ms overhead 2607.83 ms
	#~ 165.139077 GFlop/s

#~ #---------------------
