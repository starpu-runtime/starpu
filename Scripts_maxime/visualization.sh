#!/usr/bin/bash
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne 1 dynamic-data-aware
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Cholesky 1 dynamic-data-aware
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne 3 HFP
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 30 Matrice_ligne 8 HFP
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 MatriceZ4 3 HFP
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice3D 3 HFP

make -j 6
PATH_STARPU=$1
PATH_R=$2
N=$3
DOSSIER=$4
NGPU=$5
ORDO=$6
CM=500
MULTI=4
EVICTION=1
POP_POLICY=1
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 5000000

if [ $NGPU = 1 ]
	then
	MULTI=0
fi

if [ $DOSSIER = "Matrice_ligne" ]
	then
	STARPU_SCHED=${ORDO} SEED=1 EVICTION_STRATEGY_DYNAMIC_OUTER=$((EVICTION)) DATA_POP_POLICY=$((POP_POLICY)) STARPU_WORKER_STATS=1 STARPU_BUS_STATS=1 STARPU_GENERATE_TRACE=1 PRINT_IN_TERMINAL=1 PRINT_N=$((N)) STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 MULTIGPU=$((MULTI)) BELADY=0 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
	python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py ${N} ${ORDO} ${NGPU} ${DOSSIER} 1
fi
if [ $DOSSIER = "MatriceZ4" ]
	then
	STARPU_SCHED=${ORDO} PRINT_IN_TERMINAL=1 PRINT_N=$((N)) STARPU_GENERATE_TRACE=0 PRINT3D=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 MULTIGPU=$((MULTI)) TASK_STEALING=0 BELADY=0 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1
	python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py ${N} ${ORDO} ${NGPU} ${DOSSIER} 4
fi
if [ $DOSSIER = "Matrice3D" ]
	then
	STARPU_SCHED=${ORDO} PRINT_IN_TERMINAL=1 PRINT_N=$((N)) PRINT3D=2 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 MULTIGPU=4 TASK_STEALING=0 BELADY=0 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -3d -xyz $((960*N)) -nblocks $((N)) -nblocksz $((N)) -iter 1
	python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py ${N} ${ORDO} ${NGPU} ${DOSSIER} ${N}
fi
if [ $DOSSIER = "Cholesky" ]
	then
	DEPENDANCES=1 STARPU_SCHED=${ORDO} PRINT_IN_TERMINAL=1 PRINT_N=$((N)) STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=gemini-1-fgcs ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
	python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py ${N} ${ORDO} ${NGPU} ${DOSSIER}
fi

#~ python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 1
#~ python3 /home/gonthier/these_gonthier_maxime/Code/test.py Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Output_maxime/Data_to_load_SCHEDULER.txt ${N} ${ORDO} ${NGPU} 4

#~ make -C Output_maxime/

#~ if [ $NGPU = 1 ]
	#~ then
	#~ Rscript ${PATH_R}/R/ScriptR/Data_to_load.R ${DOSSIER} ${NGPU} Output_maxime/Data_to_load_GPU_0
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Data_to_load_N${N}_${NGPU}GPU_${ORDO}.pdf
#~ fi
#~ if [ $NGPU = 3 ]
	#~ then
	#~ Rscript ${PATH_R}/R/ScriptR/Data_to_load.R ${DOSSIER} ${NGPU} Output_maxime/Data_to_load_GPU_0 Output_maxime/Data_to_load_GPU_1 Output_maxime/Data_to_load_GPU_2
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/Data_to_load_N${N}_${NGPU}GPU_${ORDO}.pdf
#~ fi
