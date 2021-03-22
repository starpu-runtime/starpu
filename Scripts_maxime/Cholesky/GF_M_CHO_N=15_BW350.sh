#!/usr/bin/bash
PATH_STARPU=$1
PATH_R=$2
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
ulimit -S -s 5000000
NB_ALGO_TESTE=8
NB_TAILLE_TESTE=$3
ECHELLE_X=50
START_X=0
FICHIER=GF_M_CHO_N=15_BW350
FICHIER_RAW=${PATH_STARPU}/starpu/Output_maxime/GFlops_raw_out_2.txt
DOSSIER=Cholesky
truncate -s 0 ${FICHIER_RAW:0}
echo "########## Random ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=random_order STARPU_LIMIT_BANDWIDTH=350 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15 | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "########## Dmdar ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=dmdar STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15 | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "########## HFP ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP ORDER_U=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15 | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "########## HFP U ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15 | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "########## MST ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=mst STARPU_LIMIT_BANDWIDTH=350 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15 | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "########## CM ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=cuthillmckee STARPU_LIMIT_BANDWIDTH=350 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15 | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "########## HFP BELADY ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP ORDER_U=0 STARPU_LIMIT_BANDWIDTH=350 BELADY=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15 | tail -n 1 >> ${FICHIER_RAW:0}
done
echo "########## HFP U BELADY ##########"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((START_X+i*ECHELLE_X))
	STARPU_SCHED=HFP ORDER_U=1 STARPU_LIMIT_BANDWIDTH=350 BELADY=1 STARPU_NTASKS_THRESHOLD=30 STARPU_CUDA_PIPELINE=4 STARPU_LIMIT_CUDA_MEM=$M STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/cholesky/cholesky_implicit -size $((960*15)) -nblocks 15 | tail -n 1 >> ${FICHIER_RAW:0}
done
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW:0} ${PATH_R}/R/Data/${DOSSIER}/${FICHIER:0}.txt
Rscript ${PATH_R}/R/ScriptR/${DOSSIER}/${FICHIER:0}.R ${PATH_R}/R/Data/${DOSSIER}/${FICHIER}.txt
mv ${PATH_STARPU}/starpu/Rplots.pdf ${PATH_R}/R/Courbes/${DOSSIER}/${FICHIER:0}.pdf