#!/usr/bin/bash
#bash Scripts_maxime/PlaFRIM/GF_Workingset.sh 2 5 Matrice_ligne

NB_TAILLE_TESTE=$1
DOSSIER=$3
NB_ALGO_TESTE=2
NB_TEST_MEME_SOMMET=$2
ECHELLE_X=5
START_X=0  
FICHIER_RAW=Output_maxime/GFlops_raw_out_3.txt
module load linalg/mkl
ulimit -S -s 5000000
truncate -s 0 ${FICHIER_RAW}

if [ $DOSSIER = "Matrice_ligne" ]
	then
		echo "############## Eager ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			for ((j=1 ; j<=(($NB_TEST_MEME_SOMMET)); j++))
				do 
				STARPU_SCHED=eager STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
			done
		done
		echo "############## Dmdar ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			for ((j=1 ; j<=(($NB_TEST_MEME_SOMMET)); j++))
				do 
				STARPU_SCHED=dmdar STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
			done
		done
		echo "############## MST ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			for ((j=1 ; j<=(($NB_TEST_MEME_SOMMET)); j++))
				do 
				STARPU_SCHED=mst STARPU_NTASK_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
			done
		done
		echo "############## CM ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			for ((j=1 ; j<=(($NB_TEST_MEME_SOMMET)); j++))
				do 
				STARPU_SCHED=cuthillmckee REVERSE=1 STARPU_NTASK_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
			done
		done
		echo "############## HFP ##############"
		for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
			do 
			N=$((START_X+i*ECHELLE_X))
			for ((j=1 ; j<=(($NB_TEST_MEME_SOMMET)); j++))
				do 
				STARPU_SCHED=HFP BELADY=1 ORDER_U=1 STARPU_NTASK_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=500 STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NOPENCL=0 ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1 | tail -n 1 >> ${FICHIER_RAW}
			done
		done
fi

# Tracage des GFlops
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c -lm
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} Output_maxime/GFlops.txt
