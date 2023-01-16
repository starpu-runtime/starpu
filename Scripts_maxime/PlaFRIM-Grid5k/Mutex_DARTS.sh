#!/usr/bin/bash	
#	bash Scripts_maxime/PlaFRIM-Grid5k/Mutex_DARTS.sh 12 Matrice_ligne mutex_darts 4
#	bash Scripts_maxime/PlaFRIM-Grid5k/Mutex_DARTS.sh 3 Matrice3D mutex_darts 4
#	bash Scripts_maxime/PlaFRIM-Grid5k/Mutex_DARTS.sh 3 Cholesky mutex_darts 4
#	bash Scripts_maxime/PlaFRIM-Grid5k/Mutex_DARTS.sh 7 Cholesky mutex_darts 2

make -j 100
NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
START_X=0  
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=Output_maxime/GFlops_raw_out_3.txt
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 Output_maxime/Data/Nb_conflit_donnee.txt
truncate -s 0 Output_maxime/Data/Nb_conflit_donnee_critique.txt
truncate -s 0 ${FICHIER_BUS}
truncate -s 0 ${FICHIER_RAW_DT}
ulimit -S -s 5000000
CM=500
TH=10
CP=5
NITER=11
ECHELLE_X=$((5*NGPU))
NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))

if [ $DOSSIER = "Matrice_ligne" ]
then
	echo "############## Dynamic data aware + EVICTION ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		PRINT_TIME=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
	done
fi
if [ $DOSSIER = "Matrice3D" ]
then
	echo "############## DARTS 3D + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=1 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
	echo "############## DARTS 3D + LUF + THRESHOLD 2 + FROMMEM + SIMMEM ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=2 APP=1 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW}
	done
fi
if [ $DOSSIER = "Cholesky" ]
then
	echo "############## DARTS + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=0 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
		sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
	done
	echo "############## DARTS 3D + LUF ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=0 APP=1 CHOOSE_BEST_DATA_FROM=0 SIMULATE_MEMORY=0 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
		sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
	done
	echo "############## DARTS 3D + LUF + THRESHOLD 2 + FROMMEM + SIMMEM ##############"
	for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		do 
		N=$((START_X+i*ECHELLE_X))
		STARPU_SCHED_READY=0 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 THRESHOLD=2 APP=1 CHOOSE_BEST_DATA_FROM=1 SIMULATE_MEMORY=1 NATURAL_ORDER=0 STARPU_SCHED=dynamic-data-aware SEED=$((i)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW}
		sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
	done
fi