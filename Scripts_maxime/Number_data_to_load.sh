#!/usr/bin/bash
#~ bash Scripts_maxime/Number_data_to_load.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice_ligne 1
#~ bash Scripts_maxime/Number_data_to_load.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice_ligne 3
#~ bash Scripts_maxime/Number_data_to_load.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice3D 1

PATH_STARPU=$1
PATH_R=$2
N=$3
DOSSIER=$4
NGPU=$5
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 5000000

if [ $DOSSIER = "Matrice_ligne" ]
	then
	STARPU_SCHED=HFP PRINTF=1 STARPU_SCHED_READY=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 MULTIGPU=4 BELADY=0 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
fi
if [ $DOSSIER = "Matrice3D" ]
	then
	STARPU_SCHED=HFP PRINTF=1 STARPU_SCHED_READY=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 MULTIGPU=0 BELADY=0 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d  -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1
fi

if [ $NGPU = 1 ]
	then
	Rscript ${PATH_R}/R/ScriptR/Data_to_load.R ${DOSSIER} ${NGPU} Output_maxime/Data_to_load_GPU_0
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Data_to_load_N${N}_${NGPU}GPU.pdf
fi
if [ $NGPU = 3 ]
	then
	Rscript ${PATH_R}/R/ScriptR/Data_to_load.R ${DOSSIER} ${NGPU} Output_maxime/Data_to_load_GPU_0 Output_maxime/Data_to_load_GPU_1 Output_maxime/Data_to_load_GPU_2
	mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Data_to_load_N${N}_${NGPU}GPU.pdf
fi
