#!/usr/bin/bash
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne 1 dynamic-data-aware
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Cholesky 1 dynamic-data-aware
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 4 Matrice_ligne 1 HFP
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne 3 HFP
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 30 Matrice_ligne 8 HFP
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 MatriceZ4 3 HFP
#~ bash Scripts_maxime/visualization.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice3D 3 HFP

# Pour à la main appeller la visualisation sans lancer d'expé, par exemple si on veut modifier des choses à la main dans les données on fais: python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py 4 HFP 1 Matrice_ligne 1 1
# Les fichiers ouvert à modifier sont :
	# Output_maxime/Data_stolen_load_balance.txt Pour le task stealing pour HFP
	# Output_maxime/last_package_split.txt Pour les sous paquets avant dernier merge pour HFP
	# Output_maxime/Data_to_load_prefetch_SCHEDULER.txt Pour les chargements en prefetch pour tous
	# Output_maxime/Data_coordinates_order_last_SCHEDULER.txt Pour l'ordre de traitement des tâches pour tous
	# Output_maxime/Data_to_load_SCHEDULER.txt Pour l'ordre de chargement des données pour tous

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

# LA PERMUTATION 
# /./home/gonthier/these_gonthier_maxime/Code/permutation_visu_python $((N)) HFP 1 1 Que en 2D ? Oui :/.


if [ $DOSSIER = "Matrice_ligne" ]
	then
	STARPU_SCHED=${ORDO} SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_OUTER=0 PRINT_IN_TERMINAL=1 PRINT_N=$((N)) STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 MULTIGPU=$((MULTI)) BELADY=0 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=gemini-1-fgcs ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
	#~ /./home/gonthier/these_gonthier_maxime/Code/permutation_visu_python $((N)) DARTS 1 1
	python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py ${N} ${ORDO} ${NGPU} ${DOSSIER} 1 0
fi
if [ $DOSSIER = "MatriceZ4" ]
	then
	STARPU_SCHED=${ORDO} SEED=$((N/5)) PRINT_IN_TERMINAL=1 PRINT_N=$((N)) STARPU_GENERATE_TRACE=0 PRINT3D=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 MULTIGPU=$((MULTI)) TASK_STEALING=0 BELADY=0 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=gemini-1-fgcs ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz 4 -iter 1
	python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py ${N} ${ORDO} ${NGPU} ${DOSSIER} 4
fi
if [ $DOSSIER = "Matrice3D" ]
	then
	STARPU_SCHED=${ORDO} SEED=$((N/5)) PRINT_IN_TERMINAL=1 PRINT_N=$((N)) PRINT3D=2 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 MULTIGPU=4 TASK_STEALING=0 BELADY=0 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_BANDWIDTH=1050 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=gemini-1-fgcs ./examples/mult/sgemm -3d -xyz $((960*N)) -nblocks $((N)) -nblocksz $((N)) -iter 1
	python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py ${N} ${ORDO} ${NGPU} ${DOSSIER} ${N}
fi
if [ $DOSSIER = "Cholesky" ]
	then
	DEPENDANCES=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SCHED=${ORDO} SEED=$((N/5)) PRINT_IN_TERMINAL=1 PRINT_N=$((N)) STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=5 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_LIMIT_CUDA_MEM=500 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=gemini-1-fgcs ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
	python3 /home/gonthier/these_gonthier_maxime/Code/visualisation2D.py ${N} ${ORDO} ${NGPU} ${DOSSIER}
fi
