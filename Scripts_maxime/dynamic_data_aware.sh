#!/usr/bin/bash
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice_ligne dynamic_data_aware Attila 2
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp Attila 1
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp Attila 2

#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware gemini-2-ipdps 1 V V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware gemini-2-ipdps 2 V V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 1 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Matrice_ligne dynamic_data_aware_no_hfp gemini-2-ipdps 2 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 3 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 4 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 5 Matrice_ligne dynamic_data_aware_no_hfp gemini-1-ipdps 8 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 15 Random_task_order dynamic_data_aware_no_hfp Attila 1 V
#	bash Scripts_maxime/dynamic_data_aware.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 12 Random_task_order dynamic_data_aware_no_hfp Attila 2 V

PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
GPU=$6
NGPU=$7
START_X=0 
NITER=1
FICHIER_RAW=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_3.txt
FICHIER_TIME=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_4.txt
FICHIER_SCHEDULE_TIME_TEMP=${PATH_STARPU}/starpu/Output_maxime/Schedule_time_raw_out_temp.txt
FICHIER_SCHEDULE_TIME=${PATH_STARPU}/starpu/Output_maxime/Schedule_time_raw_out.txt
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 500000000
#~ truncate -s 0 ${FICHIER_RAW}
#~ truncate -s 0 ${FICHIER_BUS}
#~ truncate -s 0 ${FICHIER_RAW_DT}
#~ truncate -s 0 ${FICHIER_TIME}
#~ truncate -s 0 ${FICHIER_SCHEDULE_TIME_TEMP}
#~ truncate -s 0 ${FICHIER_SCHEDULE_TIME}
#~ truncate -s 0 ${PATH_STARPU}/starpu/Output_maxime/DDA_eviction_time.txt
if [ $GPU = "gemini-2-ipdps" ]
then
	HOST="gemini-2-ipdps"
fi
if [ $GPU = "gemini-1-ipdps" ]
then
	HOST="gemini-1-ipdps"
fi
if [ $GPU = "Sirocco10" ]
then
	HOST="sirocco"
	#~ BW=350 #A corriger si on utilise Sirocco
fi
if [ $GPU = "Attila" ] # Ajouter STARPU_LIMIT_BANDWIDTH si je test avec attila
then
	HOST="attila"
	BW=350
	BW=$((BW*NGPU))
fi


CM=500
#~ CM=$((CM/NGPU))

TH=10
CP=5

NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))
DECALAGE_FICHIER_SCHEDULE=$((NGPU*6+5)) # Parce que chaque GPU s'ffiche sur 6 lignes en comptant l'espace et qu'il y a 5 liges constantes

#~ STARPU_WORKER_STATS=1 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_WORKER_STATS_FILE=Output_maxime/test2.txt STARPU_BUS_STATS_FILE=Output_maxime/test1.txt STARPU_PROFILING=1

if [ $DOSSIER = "Matrice_ligne" ]
then
	NITER=1
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
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s`
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## HFPUR ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		else
			echo "Avec HFP et plus de 1 GPU"
			ECHELLE_X=$((5*NGPU))
		    NB_ALGO_TESTE=6
		    #~ echo "############## Modular eager prefetching ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ start=`date +%s`
			    #~ STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ end=`date +%s` 
				#~ echo $((end-start)) >> ${FICHIER_TIME}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    #~ sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    #~ done
		    #~ echo "############## Dmdar ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ start=`date +%s`
			    #~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ end=`date +%s` 
				#~ echo $((end-start)) >> ${FICHIER_TIME}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    #~ sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    #~ done
		    echo "############## HFPUR + load balance and task stealing TH30 ##############"
		    for ((i=8 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=1 TASK_STEALING=3 MULTIGPU=4 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    #~ echo "############## HMETIS + TASK STEALING ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt
			    #~ start=`date +%s` 
			    #~ STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ end=`date +%s` 
				#~ echo $((end-start)) >> ${FICHIER_TIME}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## Dynamic data aware TH30 Pop best data ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ start=`date +%s`
			    #~ SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ end=`date +%s` 
				#~ echo $((end-start)) >> ${FICHIER_TIME}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ start=`date +%s`
			    #~ PRINT_TIME=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ end=`date +%s` 
				#~ echo $((end-start)) >> ${FICHIER_TIME}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    #~ sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    #~ done
		fi
	fi
	if [ $MODEL = "dynamic_data_aware_no_hfp" ]
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
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		else
		    NB_ALGO_TESTE=7
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
				start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		fi
	fi
fi
if [ $DOSSIER = "Random_task_order" ]
then
	NITER=11
	if [ $MODEL = "dynamic_data_aware_no_hfp" ]
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
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		else
		    NB_ALGO_TESTE=7
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
				start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 RANDOM_TASK_ORDER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) SEED=$((i)) STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		 fi
	fi
fi

#Tracage data transfers
#~ gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
#~ ./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT:0} ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ # Tracage des GFlops
#~ gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
#~ ./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ # Tracage du temps
#~ gcc -o cut_time_raw_out cut_time_raw_out.c
#~ ./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_TIME} ${PATH_R}/R/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ # Tracage du temps global et du temps de schedule all et split
#~ gcc -o cut_schedule_time_raw_out cut_schedule_time_raw_out.c
#~ ./cut_schedule_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_SCHEDULE_TIME} ${PATH_R}/R/Data/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt SCHEDULE_TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf

#~ # A retirer surement : tracage du temps d'éviction de DDA et de schedule de DDA
#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_STARPU}/starpu/Output_maxime/DDA_eviction_time.txt EVICTION_TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/EVICTION_TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf
#~ mv ${PATH_STARPU}/starpu/Output_maxime/DDA_eviction_time.txt ${PATH_R}/R/Data/${DOSSIER}/EVICTION_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
