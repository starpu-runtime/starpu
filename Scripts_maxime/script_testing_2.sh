#!/usr/bin/bash
start=`date +%s`
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
#export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 10000000
make -C src/ -j 100
#~ sudo make -j 6

N=$1

#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1

#HOST="gemini-2"
#~ N=25
		NB_TAILLE_TESTE=10
#		echo "############## HFPU TH30 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((i*10))
			start2=`date +%s`
			SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=2 STARPU_NOPENCL=0 STARPU_BUS_STATS=1 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 11
			end2=`date +%s`
			echo $((end2-start2)) "secondes"
			done
#STARPU_SCHED=HFP HMETIS=3 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=2 STARPU_NOPENCL=0 STARPU_BUS_STATS=1 ./examples/mult/sgemm -xy $((960*100)) -nblocks 100 -iter 11
end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
