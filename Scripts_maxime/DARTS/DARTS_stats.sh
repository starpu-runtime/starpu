#!/bin/bash
#	bash Scripts_maxime/DARTS/DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Cholesky_dependances DARTS gemini-1-fgcs 1
#	bash Scripts_maxime/DARTS/DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Cholesky_dependances DATA_TASK_ORDER gemini-1-fgcs 1

#~ # Pour initialiser les fichiers de stats
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
truncate -s 0 Output_maxime/Data/DARTS/Nb_conflit_donnee.txt
truncate -s 0 Output_maxime/Data/DARTS/Nb_conflit_donnee_critique.txt

HOST=$GPU
CM=500
TH=10
CP=5
NITER=11
NCOMBINAISONS=4
if [ NGPU != 1 ]
then
	NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))
fi

if [ $DOSSIER = "Cholesky_dependances" ]
then
	if [ $MODEL = "DARTS" ]
	then
		echo "N,EAGER,DMDAR,DARTS+ LUF, DARTS + LUF + 3D,DARTS + LUF + TH2,DARTS + LUF + TH2 + 3D,DARTS + LUF + 3D + SM, DARTS + LUF + 3D + FM,DARTS + LUF + 3D + FM + SM,DARTS + LUF + 3D + FM + SM + TH2" > Output_maxime/Legende.txt
		ECHELLE_X=$((5))
		NB_ALGO_TESTE=10
		echo "############## MODULAR EAGER PREFETCHING ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DMDAR ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF + 3D ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_N=$((N)) DEPENDANCES=1 APP=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF + TH2 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_N=$((N)) DEPENDANCES=1 THRESHOLD=2 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF + TH2 + 3D ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_N=$((N)) DEPENDANCES=1 APP=1 THRESHOLD=2 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF + 3D + SM ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_N=$((N)) DEPENDANCES=1 SIMULATE_MEMORY=1 APP=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF + 3D + FM ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_N=$((N)) DEPENDANCES=1 CHOOSE_BEST_DATA_FROM=1 APP=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF + 3D + FM + SM ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_N=$((N)) DEPENDANCES=1 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 APP=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF + 3D + FM + SM + TH2 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_N=$((N)) DEPENDANCES=1 THRESHOLD=2 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 APP=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
	fi
	if [ $MODEL = "DATA_TASK_ORDER" ]
	then
		echo "N,DARTS+LUF+TO0+DO0,DARTS+LUF+TO0+DO1,DARTS+LUF+TO0+DO2,DARTS+LUF+TO1+DO0,DARTS+LUF+TO1+DO1,DARTS+LUF+TO1+DO2,DARTS+LUF+TO2+DO0,DARTS+LUF+TO2+DO1,DARTS+LUF+TO2+DO2" > Output_maxime/Legende.txt
		ECHELLE_X=$((5))
		NB_ALGO_TESTE=9
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=0 DATA_ORDER=0 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=0 DATA_ORDER=1 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=0 DATA_ORDER=2 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=1 DATA_ORDER=0 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=1 DATA_ORDER=1 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=1 DATA_ORDER=2 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=2 DATA_ORDER=0 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=2 DATA_ORDER=1 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## DARTS + LUF ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			TASK_ORDER=2 DATA_ORDER=2 PRINT_N=$((N)) DEPENDANCES=1 STARPU_HOSTNAME=${HOST} SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
	fi
fi

#~ # Tracage des GFlops
#~ gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
#~ ./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ if [[ $MODEL != "dynamic_data_aware_no_hfp_no_mem_limit" ]]
#~ then
	#Tracage data transfers
	#~ gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	#~ ./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT} ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage du temps
	#~ gcc -o cut_time_raw_out cut_time_raw_out.c
	#~ ./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_TIME} ${PATH_R}/R/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf

	# Tracage du temps global et du temps de schedule all et split
	#~ gcc -o cut_schedule_time_raw_out cut_schedule_time_raw_out.c
	#~ ./cut_schedule_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_SCHEDULE_TIME} ${PATH_R}/R/Data/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt SCHEDULE_TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf

	#~ if [[ $MODEL != "dynamic_data_aware_no_hfp_sparse_matrix" ]]
	#~ then
		#~ # A retirer surement : tracage du temps d'éviction de DDA et de schedule de DDA
		#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/DARTS_time.txt DARTS_time_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
		#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DARTS_time_${MODEL}_${GPU}_${NGPU}GPU.pdf
		#~ mv ${PATH_STARPU}/starpu/Output_maxime/DARTS_time.txt ${PATH_R}/R/Data/${DOSSIER}/DARTS_time_${MODEL}_${GPU}_${NGPU}GPU.txt
		#~ # No threshold
		#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold.txt DARTS_time_no_threshold_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
		#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DARTS_time_no_threshold_${MODEL}_${GPU}_${NGPU}GPU.pdf
		#~ mv ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold.txt ${PATH_R}/R/Data/${DOSSIER}/DARTS_time_no_threshold_${MODEL}_${GPU}_${NGPU}GPU.txt
		#~ # No threshold and choose best data from memory
		#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold_choose_best_data_from_memory.txt DARTS_time_no_threshold_choose_best_data_from_memory_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
		#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DARTS_time_no_threshold_choose_best_data_from_memory_${MODEL}_${GPU}_${NGPU}GPU.pdf
		#~ mv ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold_choose_best_data_from_memory.txt ${PATH_R}/R/Data/${DOSSIER}/DARTS_time_no_threshold_choose_best_data_from_memory_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ fi
#~ fi

# Tracage des GFlops
gcc -o cut_gflops_raw_out_csv cut_gflops_raw_out_csv.c
gcc -o cut_datatransfers_raw_out_csv cut_datatransfers_raw_out_csv.c
./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv
./cut_datatransfers_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT} ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.csv
./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X Output_maxime/Data/DARTS/Nb_conflit_donnee.txt ${PATH_R}/R/Data/${DOSSIER}/Nb_conflits_donnee_${MODEL}_${GPU}_${NGPU}GPU.csv

#~ # Plot python
#~ python3 /home/gonthier/these_gonthier_maxime/Code/Plot.py ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv
#~ mv ${PATH_STARPU}/starpu/plot.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

