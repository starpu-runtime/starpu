#!/usr/bin/bash
# ./cut_gflops_raw_out NOMBRE_DE_TAILLES_DE_MATRICES NOMBRE_ALGO_TESTE

cd..
#~ ./configure --enable-simgrid --disable-mpi --with-simgrid-dir=/home/gonthier/simgrid
#~ make install
#~ make -j4
echo "MAKE OK!"
export STARPU_PERF_MODEL_DIR=/usr/local/share/starpu/perfmodels/sampling
truncate -s 0 GFlops_raw_out.txt
ulimit -S -s 50000
NB_ALGO_TESTE=4
NB_TAILLE_TESTE=5
TAILLE=200
FICHIER=GF_M_MC_NT=225_LRU_BW=350
echo "Random"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((TAILLE*i))
	STARPU_SCHED=random STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M RANDOM_TASK_ORDER=0 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter 1 | tail -n 1 >> GFlops_raw_out.txt
done
echo "Dmdar"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((TAILLE*i))
	STARPU_SCHED=dmdar STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M RANDOM_TASK_ORDER=0 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter 1 | tail -n 1 >> GFlops_raw_out.txt
done
echo "HFP"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((TAILLE*i))
	ALGO_USED=5 STARPU_SCHED=AATO STARPU_NTASKS_THRESHOLD=30 HILBERT=0 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M RANDOM_TASK_ORDER=0 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter 1 | tail -n 1 >> GFlops_raw_out.txt
done
echo "HFP U"
for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
	do 
	M=$((TAILLE*i))
	ALGO_USED=5 STARPU_SCHED=AATO STARPU_NTASKS_THRESHOLD=30 HILBERT=1 STARPU_LIMIT_BANDWIDTH=350 STARPU_LIMIT_CUDA_MEM=$M RANDOM_TASK_ORDER=0 STARPU_DIDUSE_BARRIER=1 STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_NOPENCL=0 STARPU_HOSTNAME=attila ./examples/mult/sgemm -xy $((960*15)) -nblocks 15 -iter 1 | tail -n 1 >> GFlops_raw_out.txt
done
gcc -o cut_gflops_raw_out cut_gflops_raw_out.c
./cut_gflops_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE
mv GFlops_data_out.txt /home/gonthier/these_gonthier_maxime/Starpu/R/Data/Matrice_ligne/${FICHIER:0}.txt
Rscript /home/gonthier/these_gonthier_maxime/Starpu/R/ScriptR/Matrice_ligne/${FICHIER:0}.R
mv /home/gonthier/starpu/Rplots.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/Matrice_ligne/${FICHIER:0}.pdf
echo Fin du script
