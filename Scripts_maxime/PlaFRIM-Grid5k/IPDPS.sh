#!/usr/bin/bash	
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 1 V
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 2 V
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 10 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 3 X
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 10 Matrice_ligne dynamic_data_aware_no_hfp_ipdps 4 S

#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 13 Random_task_order dynamic_data_aware_no_hfp_ipdps 2 X

#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_ipdps_profiling 1 x
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 15 Matrice_ligne dynamic_data_aware_no_hfp_ipdps_profiling 2 x
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 10 Matrice_ligne dynamic_data_aware_no_hfp_ipdps_profiling 3 x

#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 11 Matrice_ligne dynamic_data_aware_compare_threshold 2
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 12 Matrice_ligne dynamic_data_aware_compare_threshold_type 2
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 10 Matrice_ligne dynamic_data_aware_compare_choose_best_data_type 2

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
START_X=0  
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=Output_maxime/GFlops_raw_out_3.txt
GPU=Gemini
ulimit -S -s 5000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_BUS}
truncate -s 0 ${FICHIER_RAW_DT}

CM=500

TH=10
CP=5

NITER=11

NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))

if [ $DOSSIER = "Matrice_ligne" ]
then
	if [ $MODEL = "dynamic_data_aware_no_hfp_ipdps" ]
	then
		ECHELLE_X=$((5*NGPU))
		if [ $NGPU = 1 ]
		then
			echo "NGPU = 1"
		    NB_ALGO_TESTE=6
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFPUR count 0 ##############"
		    for ((i=1 ; i<=((5)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=HFP STARPU_SCHED_READY=1 COUNT_DO_SCHEDULE=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFPUR count 1 ##############"
		    for ((i=1 ; i<=((5)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=HFP STARPU_SCHED_READY=1 COUNT_DO_SCHEDULE=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		else
		  echo "NGPU = 2, 3, 4 ou 8"
		    NB_ALGO_TESTE=5
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=HFP HMETIS=3 HMETIS_N=$((N)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		     echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
			if [ $NGPU = 4 ]
			then
				echo "############## Dynamic data aware TH30 Pop best data + EVICTION + TH ##############"
				for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
					do 
					N=$((START_X+i*ECHELLE_X))
					THRESHOLD=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
					sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
				done
			fi
		fi
	fi
	if [ $MODEL == "dynamic_data_aware_compare_threshold" ]
	then
		ECHELLE_X=20
		NB_ALGO_TESTE=1
		N=110
		echo "############## Compare Threshold ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			DDA_TH=$((START_X+i*ECHELLE_X))
			CHOOSE_BEST_DATA_THRESHOLD=$((DDA_TH)) SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
	fi
	if [ $MODEL == "dynamic_data_aware_compare_threshold_type" ]
	then
		ECHELLE_X=$((5*NGPU))
		NB_ALGO_TESTE=5
		DDA_TH=20
		echo "############## Sans rien ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			N=$((START_X+i*ECHELLE_X))
			SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Fix TH ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			N=$((START_X+i*ECHELLE_X))
			LIFT_THRESHOLD_MODE=1 CHOOSE_BEST_DATA_THRESHOLD=100 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## nb task done before lift TH ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			N=$((START_X+i*ECHELLE_X))
			LIFT_THRESHOLD_MODE=2 CHOOSE_BEST_DATA_THRESHOLD=100 NUMBER_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD=50 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## nb task done before slowly lift TH ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			N=$((START_X+i*ECHELLE_X))
			LIFT_THRESHOLD_MODE=3 CHOOSE_BEST_DATA_THRESHOLD=100 NUMBER_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD=50 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## % of task done before slowly lift TH ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			N=$((START_X+i*ECHELLE_X))
			LIFT_THRESHOLD_MODE=4 CHOOSE_BEST_DATA_THRESHOLD=100 PERCENTAGE_OF_TASK_DONE_BEFORE_LIFTING_THRESHOLD=25 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
	fi
	if [ $MODEL == "dynamic_data_aware_compare_choose_best_data_type" ]
	then
		ECHELLE_X=$((5*NGPU))
		NB_ALGO_TESTE=3
		echo "############## CHOOSE_BEST_DATA_TYPE=0 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			N=$((START_X+i*ECHELLE_X))
			CHOOSE_BEST_DATA_TYPE=0 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## CHOOSE_BEST_DATA_TYPE=1 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			N=$((START_X+i*ECHELLE_X))
			CHOOSE_BEST_DATA_TYPE=1 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## CHOOSE_BEST_DATA_TYPE=2 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do
			N=$((START_X+i*ECHELLE_X))
			CHOOSE_BEST_DATA_TYPE=2 SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
	fi
	if [ $MODEL = "dynamic_data_aware_no_hfp_ipdps_profiling" ]
	then
		ECHELLE_X=$((5*NGPU))
		if [ $NGPU = 1 ]
		then
			echo "NO HFP and NGPU = 1 profiling"
		    NB_ALGO_TESTE=5
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
			    sed -n $((DECALAGE_FICHIER_SCHEDULE))'p' ${FICHIER_SCHEDULE_TIME_TEMP} >> ${FICHIER_SCHEDULE_TIME}
		    done
		else
		  echo "NO HFP and NGPU > 1 profiling"
		    NB_ALGO_TESTE=6
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
				start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    STARPU_SCHED=HFP HMETIS=3 HMETIS_N=$((N)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_WORKER_STATS=1 STARPU_WORKER_STATS_FILE="${FICHIER_SCHEDULE_TIME_TEMP}" STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
		  echo "Random task order"
		  ECHELLE_X=5
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    RANDOM_TASK_ORDER=1 SEED=$((i)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    RANDOM_TASK_ORDER=1 SEED=$((i)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    RANDOM_TASK_ORDER=1 SEED=$((i)) STARPU_SCHED=HFP HMETIS=3 HMETIS_N=$((N)) TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		     echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    RANDOM_TASK_ORDER=1 SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    RANDOM_TASK_ORDER=1 SEED=$((i)) STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
fi

#~ #Tracage data transfers
#~ gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
#~ ./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT:0} Output_maxime/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt

#~ # Tracage des GFlops
#~ gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
#~ ./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} Output_maxime/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt

#~ # Tracage du temps
#~ gcc -o cut_time_raw_out cut_time_raw_out.c
#~ ./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_TIME} Output_maxime/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt

#~ # Tracage du temps de schedule
#~ ./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_SCHEDULE_TIME} Output_maxime/Data/${DOSSIER}/SCHEDULE_TIME_${MODEL}_${GPU}_${NGPU}GPU.txt

#~ # Penser à prendre le fichier Eviction_TIME aussi