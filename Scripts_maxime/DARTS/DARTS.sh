#!/bin/bash
#	bash Scripts_maxime/DARTS/DARTS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Cholesky_dependances DARTS gemini-1-fgcs-36 1
#	bash Scripts_maxime/DARTS/DARTS.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Cholesky_dependances DARTS gemini-1-fgcs-36 2

# Pour initialiser les fichiers de stats
#~ echo "N,Nb conflits,Nb conflits critiques" > Output_maxime/Data/DARTS/Nb_conflit_donnee.csv
#~ echo "N,Return NULL, Return task, Return NULL because main task list empty,Nb of random selection,nb_1_from_free_task_not_found" > Output_maxime/Data/DARTS/Choice_during_scheduling.csv
#~ echo "N,victim_selector_refused_not_on_node,victim_selector_refused_cant_evict,victim_selector_return_refused,victim_selector_return_unvalid,victim_selector_return_data_not_in_planned_and_pulled,victim_evicted_compteur,victim_selector_compteur,victim_selector_return_no_victim,victim_selector_belady" > Output_maxime/Data/DARTS/Choice_victim_selector.csv
#~ echo "N,Nb refused tasks,Nb new task initialized" > Output_maxime/Data/DARTS/Misc.csv
#~ echo "N,time_total_selector,time_total_evicted,time_total_belady,time_total_schedule,time_total_choose_best_data,time_total_fill_planned_task_list,time_total_initialisation,time_total_randomize, time_total_pick_random_task,time_total_least_used_data_planned_task,time_total_createtolasttaskfinished" > Output_maxime/Data/DARTS/DARTS_time.csv

make -j 100
PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
GPU=$6
NGPU=$7
START_X=0 
FICHIER_RAW=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_3.txt
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 500000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_BUS}
truncate -s 0 ${FICHIER_RAW_DT}

HOST=$GPU
CM=500
TH=10
CP=5
NITER=1
ECHELLE_X=$((5))
NCOMBINAISONS=4
if [ NGPU != 1 ]
then
	NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))
fi

if [ $DOSSIER = "Cholesky_dependances" ]
then
	if [ $MODEL = "DARTS" ]
	then
		NB_ALGO_TESTE=3
		echo "############## MODULAR EAGER PREFETCHING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DMDAR ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED_READY=1 PRIORITY_ATTRIBUTION=0 TASK_ORDER=2 DATA_ORDER=2 FREE_PUSHED_TASK_POSITION=1 DEPENDANCES=1 PRIO=1 APP=1 SEED=$((N/5)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_EXPECTED_TRANSFER_TIME_WRITEBACK=1 STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
	fi
fi

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
mv ${PATH_STARPU}/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ #Tracage data transfers
#~ gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
#~ ./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT} ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf
