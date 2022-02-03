#!/bin/bash
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 2
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp Attila 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp Attila 2

#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware gemini-2-ipdps 1 V V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware gemini-2-ipdps 2 V V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 1 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 2 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp_ipdps gemini-1-ipdps 3 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 4 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 8 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Random_task_order dynamic_data_aware_no_hfp Attila 1 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Random_task_order dynamic_data_aware_no_hfp gemini-1-ipdps 1 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 12 Random_task_order dynamic_data_aware_no_hfp Attila 2 V

#	For the testing for the rebuttal
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice3D dynamic_data_aware_no_hfp gemini-1-fgcs 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice3D dynamic_data_aware_no_hfp gemini-1-fgcs 2
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Cholesky dynamic_data_aware_no_hfp gemini-1-fgcs 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Cholesky dynamic_data_aware_no_hfp gemini-1-fgcs 2
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware_no_hfp_no_mem_limit gemini-1-fgcs 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-fgcs 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp_sparse_matrix gemini-1-fgcs 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp_sparse_matrix gemini-1-fgcs 2

#	For the testing for the actual rebuttal
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 12 Matrice3D dynamic_data_aware_no_hfp gemini-1-fgcs 4


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
#~ truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_BUS}
#~ truncate -s 0 ${FICHIER_RAW_DT}
truncate -s 0 ${PATH_STARPU}/starpu/Output_maxime/DARTS_time.txt
truncate -s 0 ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold.txt
truncate -s 0 ${PATH_STARPU}/starpu/Output_maxime/DARTS_time_no_threshold_choose_best_data_from_memory.txt
HOST=$GPU

CM=500

if [ $MODEL = "dynamic_data_aware_no_hfp_sparse_matrix" ] # Mem infinite and sparse
then
	SPARSE=10
	CM=0
fi

TH=10
CP=5

NITER=11

NCOMBINAISONS=4
if [ NGPU != 1 ]
then
	NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))
fi
#~ DECALAGE_FICHIER_SCHEDULE=$((NGPU*6+5)) # Parce que chaque GPU s'ffiche sur 6 lignes en comptant l'espace et qu'il y a 5 liges constantes

if [ $DOSSIER = "Matrice_ligne" ]
then
	if [ $MODEL = "dynamic_data_aware" ]
	then
		ECHELLE_X=5
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=7
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
				end=`date +%s`
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## HFPUR ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		else
			echo "Avec HFP et plus de 1 GPU"
			ECHELLE_X=$((5*NGPU))
		    NB_ALGO_TESTE=6
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## HFPUR + load balance and task stealing TH30 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 TASK_STEALING=3 MULTIGPU=4 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt
			    start=`date +%s` 
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    THRESHOLD=0 SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    THRESHOLD=0 PRINT_TIME=1 SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		fi
	fi
	if [[ $MODEL = "dynamic_data_aware_no_hfp_sparse_matrix" ]] || [[ $MODEL = "dynamic_data_aware_no_hfp" ]]
	then
		ECHELLE_X=$((5*NGPU))
		if [[ $MODEL = "dynamic_data_aware_no_hfp_sparse_matrix" ]]
		then
			ECHELLE_X=$((50*NGPU))
			NITER=2
		fi
		NB_ALGO_TESTE=8
		if [ $NGPU != 1 ]
		then
		    NB_ALGO_TESTE=9
		fi
		echo "############## Modular eager prefetching ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## Dmdar ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		if [ $NGPU != 1 ]
		then
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		fi
		echo "############## Dynamic data aware ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## Dynamic data aware + EVICTION ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_TIME=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## Dynamic data aware + EVICTION + TH ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_TIME=1 THRESHOLD=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## Dynamic data aware + EVICTION + SIMULATE MEMORY 1 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SIMULATE_MEMORY=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## Dynamic data aware + EVICTION + CHOOSE_BEST_DATA_FROM 1 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			PRINT_TIME=1 CHOOSE_BEST_DATA_FROM=1 APP=0 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
		echo "############## Dynamic data aware + EVICTION + CHOOSE_BEST_DATA_FROM 1 + SIMULATE_MEMORY=1 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SIMULATE_MEMORY=1 CHOOSE_BEST_DATA_FROM=1 APP=0 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		done
	fi
	if [ $MODEL = "dynamic_data_aware_no_hfp_no_mem_limit" ]
	then
		TAILLE_TUILE=1920
		NITER=1
		ECHELLE_X=$((100))
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=4
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((TAILLE_TUILE*N)) -z $((TAILLE_TUILE*4)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		    done
		fi
	fi
fi
if [ $DOSSIER = "Random_task_order" ]
then
	if [[ $MODEL = "dynamic_data_aware_no_hfp" ]] || [[ $MODEL = "dynamic_data_aware_no_hfp_sparse_matrix" ]]
	then
		ECHELLE_X=$((5*NGPU))
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=6
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		else
		    NB_ALGO_TESTE=7
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
				start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		 fi
	fi
fi
if [ $DOSSIER = "Matrice3D" ]
then
	NB_ALGO_TESTE=4
	ECHELLE_X=10
	if [ $NGPU != 1 ]
	then
		NB_ALGO_TESTE=5
	fi
	echo "############## Modular eager prefetching ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_HOSTNAME=${HOST} STARPU_SCHED=modular-eager-prefetching SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
	done
	echo "############## Dmdar ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_HOSTNAME=${HOST} STARPU_SCHED=dmdar SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
	done
	if [ $NGPU != 1 ]
	then
	echo "############## HMETIS + TASK STEALING ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
		STARPU_SCHED_READY=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_HOSTNAME=${HOST} STARPU_SCHED=HFP SEED=$((i)) HMETIS_N=$((N)) HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
	done
	fi
	echo "############## DARTS + LUF ##############"
	for ((i=13 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=0 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
	done
	echo "############## DARTS 3D + LUF ##############"
	for ((i=14 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=1 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" STARPU_HOSTNAME=${HOST} STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
		sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
	done
fi
if [ $DOSSIER = "Cholesky" ]
then
	NITER=11
	if [[ $MODEL = "dynamic_data_aware_no_hfp" ]] || [[ $MODEL = "dynamic_data_aware_no_hfp_sparse_matrix" ]]
	then
		ECHELLE_X=$((5*NGPU))
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=11
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + TH ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    THRESHOLD=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    PRINT_TIME=2 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + TH ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    PRINT_TIME=2 THRESHOLD=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + TH + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    THRESHOLD=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + SIMULATE MEMORY 1 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SIMULATE_MEMORY=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + CHOOSE_BEST_DATA_FROM 1 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    PRINT_TIME=2 CHOOSE_BEST_DATA_FROM=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + CHOOSE_BEST_DATA_FROM 1 + SIMULATE_MEMORY=1 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SIMULATE_MEMORY=1 CHOOSE_BEST_DATA_FROM=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		else
			NB_ALGO_TESTE=12
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + TH ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    THRESHOLD=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    PRINT_TIME=2 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + TH ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    PRINT_TIME=2 THRESHOLD=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + TH + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    THRESHOLD=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + SIMULATE MEMORY 1 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SIMULATE_MEMORY=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + CHOOSE_BEST_DATA_FROM 1 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    PRINT_TIME=2 CHOOSE_BEST_DATA_FROM=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		    echo "############## Dynamic data aware + EVICTION + 3D + CHOOSE_BEST_DATA_FROM 1 + SIMULATE_MEMORY=1 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SIMULATE_MEMORY=1 CHOOSE_BEST_DATA_FROM=1 APP=1 STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_HOSTNAME=${HOST} SPARSE_MATRIX=$((SPARSE)) SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
			     sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS} >> ${FICHIER_RAW_DT}
		    done
		fi
	fi
fi

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

if [[ $MODEL != "dynamic_data_aware_no_hfp_no_mem_limit" ]]
then
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
fi
