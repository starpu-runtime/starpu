#!/usr/bin/bash
#bash Scripts_maxime/mult_mct.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ 10 Matrice_ligne Multiplier

PATH_STARPU=$1
PATH_R=$2
NB_TAILLE_TESTE=$3
DOSSIER=$4
MODEL=$5
NB_ALGO_TESTE=3
FICHIER_RAW=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_1.txt
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 5000000
truncate -s 0 ${FICHIER_RAW}
if [ $DOSSIER = "Matrice_ligne" ]
	then
		START_X=99
		ECHELLE_X=1
		echo "############## Modular heft + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_NTASKS_THRESHOLD=30 MCTMULTIPLIER=$((N)) MULTIGPU=0 MODULAR_HEFT_HFP_MODE=1 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=modular-heft-HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*35)) -nblocks 35 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## modular heft idle + HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_NTASKS_THRESHOLD=30 MCTMULTIPLIER=$((N)) MULTIGPU=0 MODULAR_HEFT_HFP_MODE=2 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=modular-heft-HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*35)) -nblocks 35 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
		echo "############## HFP + load balance ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			STARPU_NTASKS_THRESHOLD=30 MULTIGPU=4 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_CUDA_PIPELINE=4 STARPU_SCHED=HFP STARPU_SIMGRID_CUDA_MALLOC_COST=0 ORDER_U=1 STARPU_LIMIT_BANDWIDTH=1050 STARPU_LIMIT_CUDA_MEM=250 STARPU_NCPU=0 STARPU_NCUDA=3 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*35)) -nblocks 35 -iter 1 | tail -n 1 >> ${FICHIER_RAW}
		done
fi
#~ if [ $DOSSIER = "Matrice3D" ]
	#~ then
#~ fi
# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}.txt
Rscript ${PATH_R}/R/ScriptR/GF_X.R ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}.txt ${MODEL} ${DOSSIER}
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/GF_${MODEL}.pdf
