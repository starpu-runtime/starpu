#!/usr/bin/bash
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne HFP gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 13 Matrice_ligne HFP gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 16 Matrice_ligne HFP gemini-1-fgcs 1

#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne HFP_memory gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne HFP_memory gemini-1-fgcs 1

#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice3D HFP gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 8 Matrice3D HFP gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice3D HFP gemini-1-fgcs 1

#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Cholesky HFP gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 7 Cholesky HFP gemini-1-fgcs 1

#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Random_tasks HFP gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 13 Random_tasks HFP gemini-1-fgcs 1

#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Random_task_order HFP gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 16 Random_task_order HFP gemini-1-fgcs 1

#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Random_set_of_task HFP gemini-1-fgcs 1
#	bash Scripts_maxime/HFP.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Random_set_of_task HFP gemini-1-fgcs 1

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
FICHIER_TIME=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_4.txt
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 500000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_BUS}
truncate -s 0 ${FICHIER_RAW_DT}
truncate -s 0 ${FICHIER_TIME}
truncate -s 0 ${PATH_STARPU}/starpu/Output_maxime/HFP_time.txt

HOST=$GPU

CM=500
TH=10
CP=5
NITER=2 # 2 pour ignorer la première. Ensuite elle sont toutes identiques

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

if [ $DOSSIER = "Matrice_ligne" ] || [ $DOSSIER = "Random_tasks" ] || [ $DOSSIER = "Random_task_order" ]
then
	if [ $MODEL = "HFP" ]
	then
		ECHELLE_X=5
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=8
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s`
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 FASTER_FIRST_ITERATION=1 STARPU_SCHED=HFP RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=mst RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=cuthillmckee RANDOM_DATA_ACCESS=$((RANDOMDATA)) RANDOM_TASK_ORDER=$((RANDOMORDER)) REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
	if [ $MODEL = "HFP_memory" ]
	then
		ECHELLE_X=50
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=8
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## HFP + U ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=HFP STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## HFP + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## HFP + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=HFP STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## HFP + U + R + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    PRINT_TIME=1 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((N)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
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
		    NB_ALGO_TESTE=8
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s`
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP FASTER_FIRST_ITERATION=1 STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 FASTER_FIRST_ITERATION=1 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter $((NITER)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
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
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s`
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N)) | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
fi
if [ $DOSSIER = "Random_set_of_task" ]
then
	if [ $MODEL = "HFP" ]
	then
		ECHELLE_X=50
		if [ $NGPU = 1 ]
		then
		    NB_ALGO_TESTE=8
		    echo "############## Modular eager prefetching ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=modular-eager-prefetching STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/random_task_graph/random_task_graph -ntasks $((N)) -ndata 400 -degreemax 2 | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## Dmdar ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/random_task_graph/random_task_graph -ntasks $((N)) -ndata 40 -degreemax 5 | tail -n 1 >> ${FICHIER_RAW:0}
				end=`date +%s`
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=0 BELADY=0 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/random_task_graph/random_task_graph -ntasks $((N)) -ndata 40 -degreemax 5 | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + R ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=0 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/random_task_graph/random_task_graph -ntasks $((N)) -ndata 40 -degreemax 5 | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=HFP STARPU_SCHED_READY=0 BELADY=1 ORDER_U=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/random_task_graph/random_task_graph -ntasks $((N)) -ndata 40 -degreemax 5 | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## HFP + U + R + BELADY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    PRINT_TIME=1 STARPU_SCHED=HFP STARPU_SCHED_READY=1 BELADY=1 ORDER_U=1 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/random_task_graph/random_task_graph -ntasks $((N)) -ndata 40 -degreemax 5 | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## MST ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=mst STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/random_task_graph/random_task_graph -ntasks $((N)) -ndata 40 -degreemax 5 | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		    echo "############## RCM ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    start=`date +%s`
			    STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_SCHED_READY=0 STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/random_task_graph/random_task_graph -ntasks $((N)) -ndata 40 -degreemax 5 | tail -n 1 >> ${FICHIER_RAW:0}
			    end=`date +%s` 
				echo $((end-start)) >> ${FICHIER_TIME}
			    sed -n '4,'$((NCOMBINAISONS))'p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		    done
		fi
	fi
fi

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
mv ~/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf

if [ $MODEL != "HFP_memory" ]
then
	# Tracage data transfers
	gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
	./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT:0} ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt
	Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt DT_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	mv ~/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.pdf

	#~ # Tracage du temps
	#~ gcc -o cut_time_raw_out cut_time_raw_out.c
	#~ ./cut_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_TIME} ${PATH_R}/R/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt
	#~ Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.txt TIME_${MODEL} ${DOSSIER} ${GPU} ${NGPU} ${NITER}
	#~ mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/TIME_${MODEL}_${GPU}_${NGPU}GPU.pdf
fi
