#!/usr/bin/bash
start=`date +%s`
ulimit -S -s 5000000
#~ export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

N=0

NGPU=1

ORDO="dmdar"

CM=500

EVICTION=0

READY=1

TH=10

CP=5

HOST="gemini-1-fgcs"

SEED=1

PRINTF=0

BELADY=0

MULTI=0

STEALING=0

NITER=1

NB_TAILLE_TESTE=8

THRESHOLD_DMDAR=4000

FICHIER=Output_maxime/test.txt
truncate -s 0 ${FICHIER}

for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do
	N=$((5*i))
	DMDAR_THRESHOLD=$((THRESHOLD_DMDAR)) TASK_STEALING=$((STEALING)) STARPU_BUS_STATS=1 MULTIGPU=$((MULTI)) PRINTF=$((PRINTF)) SEED=$((SEED)) STARPU_SCHED=${ORDO} BELADY=$((BELADY)) ORDER_U=0 STARPU_SCHED_READY=$((READY)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1 | tail -n 1 >> ${FICHIER}
done

echo "Résultats pour" ${ORDO}
cat ${FICHIER}

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
