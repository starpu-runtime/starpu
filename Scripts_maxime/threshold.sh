#!/usr/bin/bash
#	bash Scripts_maxime/threshold.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 30 Matrice_ligne threshold Attila 1
#	bash Scripts_maxime/threshold.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 30 Matrice_ligne cuda_pipeline Attila 1

PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
GPU=$6
NGPU=$7
START_X=0  
FICHIER_RAW=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_1.txt
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 5000000
truncate -s 0 ${FICHIER_RAW}
if [ $GPU = "Gemini02" ]
then
	HOST="gemini-2"
fi
if [ $GPU = "Sirocco10" ]
then
	HOST="sirocco"
fi
if [ $GPU = "Attila" ]
then
	HOST="attila"
fi

BW=350
BW=$((BW*NGPU))

CM=500

if [ $DOSSIER = "Matrice_ligne" ]
then
	if [ $MODEL = "threshold" ]
	then
		ECHELLE_X=1
		NB_ALGO_TESTE=5
		echo "############## Modular eager prefetching ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		    do 
		    N=$((START_X+i*ECHELLE_X))
		    STARPU_SCHED=modular-eager-prefetching STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=$((N)) STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
		done
		echo "############## Dmdar ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
		    do 
		    N=$((START_X+i*ECHELLE_X))
		    STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=$((N)) STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
		done
		    echo "############## Dynamic data aware TH30 Pop best data ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=$((N)) STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=$((N)) STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
		    done
		    echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
		    for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			    do 
			    N=$((START_X+i*ECHELLE_X))
			    SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=$((N)) STARPU_CUDA_PIPELINE=5 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
		    done
	fi
	if [ $MODEL = "cuda_pipeline" ]
	then
			ECHELLE_X=1
			NB_ALGO_TESTE=5
			echo "############## Modular eager prefetching ##############"
			for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
				do 
				N=$((START_X+i*ECHELLE_X))
				STARPU_SCHED=modular-eager-prefetching STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=$((N)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			done
			echo "############## Dmdar ##############"
			for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
				do 
				N=$((START_X+i*ECHELLE_X))
				STARPU_SCHED=dmdar STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=$((N)) STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			done
				echo "############## Dynamic data aware TH30 Pop best data ##############"
				for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
					do 
					N=$((START_X+i*ECHELLE_X))
					SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=$((N)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
				done
				echo "############## Dynamic data aware TH30 Pop best data + READY ##############"
				for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
					do 
					N=$((START_X+i*ECHELLE_X))
					SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=$((N)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
				done
				echo "############## Dynamic data aware TH30 Pop best data + EVICTION ##############"
				for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
					do 
					N=$((START_X+i*ECHELLE_X))
					SEED=0 STARPU_SCHED=dynamic-data-aware STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=1 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=10 STARPU_CUDA_PIPELINE=$((N)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*60)) -nblocks $((60)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
				done
		fi
fi

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.pdf
