#!/usr/bin/bash
#	bash Scripts_maxime/dynamic_outer.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne dynamic_outer Attila 1
#	bash Scripts_maxime/dynamic_outer.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 2 Matrice_ligne dynamic_outer Attila 4

PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
GPU=$6
NGPU=$7
ECHELLE_X=5
START_X=0  
FICHIER_RAW=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_1.txt
FICHIER_BUS=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_2.txt
FICHIER_RAW_DT=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_3.txt
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling
ulimit -S -s 5000000
truncate -s 0 ${FICHIER_RAW}
truncate -s 0 ${FICHIER_BUS}
truncate -s 0 ${FICHIER_RAW_DT}
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
CM=$((CM/NGPU))
if [ $NGPU = 1 ]
then
	MULTI=0
else
	MULTI=4
fi

if [ $DOSSIER = "Matrice_ligne" ]
then
	if [ $MODEL = "dynamic_outer" ]
	then
		NB_ALGO_TESTE=8
		echo "############## Modular eager prefetching ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED=modular-eager-prefetching STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dmdar ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_SCHED=dmdar STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		#~ echo "############## HFPUR TH30 ##############"
		#~ for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			#~ do 
			#~ N=$((START_X+i*ECHELLE_X))
			#~ STARPU_SCHED=HFP STARPU_SCHED_READY=1 MULTIGPU=$((MULTI)) STARPU_LIMIT_BANDWIDTH=$((BW)) ORDER_U=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			#~ sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		#~ done
		echo "############## Dynamic outer TH30 ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SEED=0 STARPU_SCHED=dynamic-outer STARPU_SCHED_READY=0 DATA_POP_POLICY=0 EVICTION_STRATEGY_DYNAMIC_OUTER=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic outer TH30 READY ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SEED=0 STARPU_SCHED=dynamic-outer STARPU_SCHED_READY=1 DATA_POP_POLICY=0 EVICTION_STRATEGY_DYNAMIC_OUTER=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic outer TH30 Pop best data ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SEED=0 STARPU_SCHED=dynamic-outer STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_OUTER=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic outer TH30 Pop best data + READY ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SEED=0 STARPU_SCHED=dynamic-outer STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_OUTER=0 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic outer TH30 Pop best data + EVICTION ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SEED=0 STARPU_SCHED=dynamic-outer STARPU_SCHED_READY=0 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_OUTER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
		echo "############## Dynamic outer TH30 Pop best data + EVICTION + READY ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			SEED=0 STARPU_SCHED=dynamic-outer STARPU_SCHED_READY=1 DATA_POP_POLICY=1 EVICTION_STRATEGY_DYNAMIC_OUTER=1 STARPU_LIMIT_BANDWIDTH=$((BW)) STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=30 STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} STARPU_BUS_STATS=1 STARPU_BUS_STATS_FILE="${FICHIER_BUS:0}" ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW:0}
			sed -n '4p' ${FICHIER_BUS:0} >> ${FICHIER_RAW_DT:0}
		done
	fi
fi

#Tracage data transfers
gcc -o cut_datatransfers_raw_out cut_datatransfers_raw_out.c
./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW_DT:0} ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/DT_${MODEL}_${GPU}.txt DT_${MODEL} ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/DT_${MODEL}_${GPU}.pdf

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}.txt ${MODEL} ${DOSSIER} ${GPU} ${NGPU}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}_${GPU}.pdf
