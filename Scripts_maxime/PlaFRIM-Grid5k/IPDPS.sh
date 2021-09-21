#!/usr/bin/bash
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPDS.sh 2 Matrice_ligne dynamic_data_aware 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/IPDPS.sh 2 Matrice_ligne dynamic_data_aware_no_hfp 1

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
START_X=0  
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=Output_maxime/GFlops_raw_out_3.txt
FICHIER_TIME=Output_maxime/GFlops_raw_out_4.txt
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 5000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_BUS}
truncate -s 0 ${FICHIER_RAW_DT}
truncate -s 0 ${FICHIER_TIME}

#~ BW=350
#~ BW=$((BW*NGPU))

CM=500
#~ CM=$((CM/NGPU))

NITER = 1
#~ Il faut mettre 11 pour 10 iération car la première est ignoré

if [ $DOSSIER = "Matrice_ligne" ]
then
	if [ $MODEL = "dynamic_data_aware" ]
	then
		ECHELLE_X=5
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=5
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
			 echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		else
		    NB_ALGO_TESTE=8
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFPUR + load balance and task stealing TH30 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=1 TASK_STEALING=3 MULTIGPU=4 ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HMETIS + HFP + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_SCHED=HFP HMETIS=2 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
	if [ $MODEL = "dynamic_data_aware_no_hfp" ]
	then
		ECHELLE_X=$((5*NGPU))
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=5
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		else
		    NB_ALGO_TESTE=6
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
				start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HMETIS + TASK STEALING ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    echo $((NGPU)) "1 20 1 1 2 0 0" > Output_maxime/hMETIS_parameters.txt 
			    STARPU_SCHED=HFP HMETIS=1 TASK_STEALING=3 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 ORDER_U=1 STARPU_SIMGRID_CUDA_MALLOC_COST=0 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
fi

#Tracage data transfers
gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW_DT:0} Output_maxime/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} Output_maxime/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt

# Tracage du temps
gcc -o cut_time_raw_out cut_time_raw_out.c
./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_TIME} Output_maxime/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt