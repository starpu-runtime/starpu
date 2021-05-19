#!/usr/bin/bash
start=`date +%s`
#~ export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 50000000
sudo make -C src/ -j 6
#~ sudo make -j 6

#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1

HOST="gemini-2"
#~ N=25
		NB_TAILLE_TESTE=9
		echo "############## HFPU TH30 ##############"
		for ((i=8 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((i*5))
			STARPU_SCHED=HFP ORDER_U=1 READY=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1 
		done
end=`date +%s`
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
