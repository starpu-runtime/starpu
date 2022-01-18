#!/usr/bin/bash
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 2 Matrice_ligne HFP 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 18 Matrice_ligne HFP 1

#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 2 Matrice_ligne HFP_memory 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Matrice_ligne HFP_memory 1

#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 2 Matrice3D HFP 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Matrice3D HFP 1

#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 2 Cholesky HFP 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 10 Cholesky HFP 1

#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 2 Random_tasks HFP 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Random_tasks HFP 1

#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 2 Random_task_order HFP 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Random_task_order HFP 1

#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 2 Sparse HFP 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Sparse HFP 1

#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 2 Sparse HFP_mem_infinie 1
#	bash Scripts_maxime/PlaFRIM-Grid5k/FGCS.sh 16 Sparse HFP_mem_infinie 1

NB_TAILLE_TESTE=$1
DOSSIER=$2
MODEL=$3
NGPU=$4
START_X=0 
FICHIER_RAW=Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=Output_maxime/GFlops_raw_out_3.txt
ulimit -S -s 500000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_BUS}
truncate -s 0 ${FICHIER_RAW_DT}

CM=500
TH=10
CP=5
NITER=11

NCOMBINAISONS=$((NGPU*2+(NGPU-1)*NGPU+3))

RANDOMDATA=0
RANDOMORDER=0
if [ $DOSSIER = "Random_tasks" ]
then
	RANDOMDATA=1
fi
if [ $DOSSIER = "Random_task_order" ]
then
	RANDOMORDER=1
fi

if [ $DOSSIER = "Matrice_ligne" ]
then
	if [ $MODEL = "HFP" ]
	then
		ECHELLE_X=5
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=10
		    #~ echo "############## Modular eager prefetching ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## Dmdar ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + R ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY COUNT 1 ##############"
		    #~ for ((i=17 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY COUNT 1 FAST FIRST ITERATION ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
	if [ $MODEL = "HFP_memory" ]
	then
		ECHELLE_X=50
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=10
		    #~ echo "############## Modular eager prefetching ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    #~ done
		    #~ echo "############## Dmdar ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    #~ done
		    #~ echo "############## HFP + U ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    #~ done
		    #~ echo "############## HFP + R ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    #~ done
		    #~ echo "############## HFP + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY COUNT 1 ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY COUNT 1 FAST FIRST ITERATION ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    #~ done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		fi
	fi
fi
if [ $DOSSIER = "Random_tasks" ] || [ $DOSSIER = "Random_task_order" ]
then
	if [ $MODEL = "HFP" ]
	then
		ECHELLE_X=5
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=8
		    #~ echo "############## Modular eager prefetching ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ SEED=$((i)) STARPU_SCHED=modular-eager-prefetching RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## Dmdar ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ SEED=$((i)) STARPU_SCHED=dmdar RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				#~ end=`date +%s`
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 SEED=$((i)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + R ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 SEED=$((i)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 SEED=$((i)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 SEED=$((i)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SEED=$((i)) COUNT_DO_SCHEDULE=0 STARPU_SCHED=mst RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SEED=$((i)) COUNT_DO_SCHEDULE=0 STARPU_SCHED=cuthillmckee RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
fi
if [ $DOSSIER = "Matrice3D" ]
then
	if [ $MODEL = "HFP" ]
	then
		ECHELLE_X=5
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=10
		    #~ echo "############## Modular eager prefetching ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## Dmdar ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				#~ end=`date +%s`
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + R ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY COUNT 1 ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY COUNT 1 FAST FIRST ITERATION ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=1 STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
fi
if [ $DOSSIER = "Cholesky" ]
then
	if [ $MODEL = "HFP" ]
	then
		ECHELLE_X=5
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=8
		    #~ echo "############## Modular eager prefetching ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## Dmdar ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
				#~ end=`date +%s`
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + R ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    #~ echo "############## HFP + U + R + BELADY ##############"
		    #~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    #~ do 
			    #~ N=$((START_X+i*ECHELLE_X))
			    #~ COUNT_DO_SCHEDULE=0 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    #~ sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    #~ done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
fi
if [ $DOSSIER = "Sparse" ]
then
	SPARSE=2
	ECHELLE_X=$((50))
	if [ $MODEL = "HFP" ]
	then
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=10
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY COUNT 1 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=1 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY COUNT 1 FAST FIRST ITERATION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=1 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) COUNT_DO_SCHEDULE=0 STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) COUNT_DO_SCHEDULE=0 STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
	if [ $MODEL = "HFP_mem_infinie" ]
	then
		CM=0
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=10
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=0 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY COUNT 1 ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=1 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=0 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY COUNT 1 FAST FIRST ITERATION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    COUNT_DO_SCHEDULE=1 SPARSE_MATRIX=$((SPARSE)) STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) COUNT_DO_SCHEDULE=0 STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SPARSE_MATRIX=$((SPARSE)) COUNT_DO_SCHEDULE=0 STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
fi
